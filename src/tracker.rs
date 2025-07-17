use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

// Length is a possibly infinite size
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
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

impl Ord for Length {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Length::Finite(a), Length::Finite(b)) => a.cmp(b),
            (Length::Infinite, Length::Infinite) => Ordering::Equal,
            (Length::Infinite, _) => Ordering::Greater,
            (_, Length::Infinite) => Ordering::Less,
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
    // Generate a vector of samples up to `desired` length. `position` indicates where
    // the beginning of the result is relative to the start of the waveform.
    fn generate(&self, sample_frequency: i32, position: usize, desired: usize) -> Vec<f32> {
        match self {
            Waveform::SineWave { frequency } => {
                let mut out = vec![0.0; desired];
                for (i, f) in out.iter_mut().enumerate() {
                    let t_secs = (i + position) as f32 / sample_frequency as f32;
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
                    *x = initial_value + slope * ((i + position) as f32 / sample_frequency as f32);
                }
                return out;
            }
            Waveform::Fin { duration, waveform } => {
                let length = (duration * sample_frequency as f32) as usize;
                if position >= length {
                    return Vec::new(); // No samples to generate
                }
                return waveform.generate(
                    sample_frequency,
                    position,
                    desired.min(length - position),
                );
            }
            Waveform::Seq { waveform, .. } => {
                return waveform.generate(sample_frequency, position, desired);
            }
            Waveform::Sum(a, b) => {
                generate_binary_op(|x, y| x + y, a, b, sample_frequency, position, desired)
            }
            Waveform::DotProduct(a, b) => {
                // Like sum, but we need to make sure we generate a length based on
                // the shorter waveform.
                let new_desired = match self.length(sample_frequency) {
                    Length::Finite(length) => {
                        if length > position {
                            desired.min(length - position)
                        } else {
                            return Vec::new(); // No samples to generate
                        }
                    }
                    Length::Infinite => desired,
                };
                generate_binary_op(|x, y| x * y, a, b, sample_frequency, position, new_desired)
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
                a.length(sample_frequency).max(length)
            }
            Waveform::DotProduct(a, b) => {
                let length =
                    Length::Finite(a.offset(sample_frequency)) + b.length(sample_frequency);
                a.length(sample_frequency).min(length)
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
    position: usize,
    desired: usize,
) -> Vec<f32> {
    let mut left = a.generate(sample_frequency, position, desired);

    let right_offset = a.offset(sample_frequency);
    if right_offset >= position + desired {
        // Make sure the left side is long enough so that we get another chance to
        // generate the right-hand side.
        left.resize(desired, 0.0);
    } else {
        // right_offset < position + desired
        // There is an overlap between the desired portion and the right waveform...
        //    1) ... and the right waveform starts after position
        // or 2) ... and the right waveform starts before position

        if position + left.len() < right_offset {
            // Either way, if the left side is shorter than the next offset, than extend it.
            left.resize(right_offset - position, 0.0);
        }

        if position < right_offset {
            // ... and the right waveform starts after position
            let right = b.generate(sample_frequency, 0, desired - (right_offset - position));
            // Merge the overlapping portion
            for (i, x) in left[right_offset - position..].iter_mut().enumerate() {
                if i >= right.len() {
                    break;
                }
                *x = op(*x, right[i]);
            }
            // If the left side is shorter than the right, than append.
            if right.len() + right_offset > left.len() + position {
                left.extend_from_slice(&right[(left.len() + position - right_offset)..]);
            }
        } else {
            // ... and the right waveform starts before  position
            let right = b.generate(sample_frequency, position - right_offset, desired);
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

pub enum Command {
    PlayOnce {
        // A unique id for this waveform
        id: u32,
        waveform: Waveform,
        // When the waveform should start playing, in beats from the beginning of playback
        at_beat: u64,
    },
    SendCurrentBuffer,
}

#[derive(Debug, Clone)]
pub struct ActiveWaveform {
    pub id: u32,
    pub waveform: Waveform,
    pub position: usize,
}

#[derive(Debug, Clone)]
pub struct PendingWaveform {
    pub id: u32,
    pub waveform: Waveform,
    pub beat: u64, // The beat at which this waveform should be played
}

#[derive(Debug, Clone)]
pub struct Status {
    pub active_waveforms: Vec<ActiveWaveform>,
    pub pending_waveforms: Vec<PendingWaveform>,
    pub current_beat: u64,
    pub samples: Option<Vec<f32>>,
}

pub struct Tracker {
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    status_sender: Sender<Status>,

    // Internal state
    active_waveforms: Vec<ActiveWaveform>,
    pending_waveforms: Vec<PendingWaveform>,
    current_beat: u64,
    samples_to_next_beat: usize,
}

pub fn new_tracker(
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    status_sender: Sender<Status>,
) -> Tracker {
    return Tracker {
        sample_frequency,
        beats_per_minute,
        command_receiver,
        status_sender,

        active_waveforms: Vec::new(),
        pending_waveforms: Vec::new(),
        current_beat: 1,
        samples_to_next_beat: samples_per_beat(sample_frequency, beats_per_minute),
    };
}

fn samples_per_beat(sample_frequency: i32, beats_per_minute: i32) -> usize {
    // (seconds/minute) * 1/(beats/min) * (samples/sec)
    let seconds_per_beat = 60.0 / beats_per_minute as f32;
    (sample_frequency as f32 * seconds_per_beat) as usize
}

// TODO add metrics

use std::sync::mpsc::TryRecvError;

impl<'a> AudioCallback for Tracker {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        // Check to see if we have any new commands
        match self.command_receiver.try_recv() {
            Ok(Command::PlayOnce {
                id,
                waveform,
                at_beat,
            }) => {
                println!(
                    "Received command to play once at beat {} with waveform {}: {:?}",
                    at_beat, id, waveform
                );
                if at_beat < self.current_beat {
                    println!(
                        "Ignoring command to play waveform {} at beat {} (current beat is {})",
                        id, at_beat, self.current_beat
                    );
                } else {
                    self.pending_waveforms.push(PendingWaveform {
                        id,
                        waveform,
                        beat: at_beat,
                    });
                }
            }
            Ok(Command::SendCurrentBuffer) => {
                println!("Received command to send current buffer");
            }
            Err(TryRecvError::Empty) => {}
            Err(e) => {
                println!("Error receiving command: {:?}", e);
            }
        }

        // Now generate!
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut filled = 0; // How much of the out buffer we've filled so far
        while filled < out.len() {
            let mut desired: usize = out.len() - filled;
            if self.samples_to_next_beat == 0 {
                self.samples_to_next_beat =
                    samples_per_beat(self.sample_frequency, self.beats_per_minute);
                self.current_beat += 1;
                println!("Advancing to beat {}", self.current_beat);

                // If we are at the start of a beat, check to see if there any any pending waveforms
                // that can become active waveforms.
                let mut i = 0;
                while i < self.pending_waveforms.len() {
                    let pending = &self.pending_waveforms[i];
                    if pending.beat == self.current_beat {
                        // This waveform can become active
                        self.active_waveforms.push(ActiveWaveform {
                            id: pending.id,
                            waveform: pending.waveform.clone(),
                            position: 0,
                        });
                        println!(
                            "Activating waveform {} at beat {}",
                            pending.id, self.current_beat
                        );
                        self.pending_waveforms.remove(i);
                    } else {
                        i += 1; // Only remove if we activated it
                    }
                }
            }

            // Only generate samples up to the next beat
            desired = desired.min(self.samples_to_next_beat);
            self.samples_to_next_beat -= desired;

            // Finally, walk through the waveforms and generate samples up to the next beat. If there
            // are no active waveforms, then just updated filled and continue.
            if self.active_waveforms.len() == 0 {
                filled += desired;
                continue;
            }

            let mut i = 0;
            while i < self.active_waveforms.len() {
                let active = &mut self.active_waveforms[i];
                let tmp = active
                    .waveform
                    .generate(self.sample_frequency, active.position, desired);
                if tmp.len() > desired {
                    panic!(
                        "Generated more samples than desired: {} > {} for waveform id {} at position {}: {:?}", 
                        tmp.len(), desired, active.id, active.position, active.waveform);
                }
                if i == 0 {
                    // If this is the first, just overwrite the out buffer
                    (out[filled..filled + tmp.len()]).copy_from_slice(&tmp);
                } else {
                    // If this is not the first waveform, then we need to add the samples to the out buffer
                    for (j, x) in tmp.iter().enumerate() {
                        out[filled + j] += *x;
                    }
                }
                if tmp.len() < desired {
                    // If we didn't generate enough samples, then remove this waveform from the active list
                    println!(
                        "Removing waveform {} at position {}",
                        active.id, active.position
                    );
                    self.active_waveforms.remove(i);
                } else {
                    active.position += desired;
                    i += 1;
                }
            }
            filled += desired;
        }

        /*
        if self.send_counter == 0 {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            self.sample_sender.send(copy).unwrap();
            self.send_counter = 5;
        } else {
            self.send_counter -= 1;
        }
        */
        self.status_sender
            .send(Status {
                active_waveforms: self.active_waveforms.clone(),
                pending_waveforms: self.pending_waveforms.clone(),
                current_beat: self.current_beat,
                samples: None,
            })
            .unwrap();
    }
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

        // Test a case to make sure that the sum generates enough samples, even when
        // the left-hand side is shorter and the right hasn't started yet.
        let mut result = w5.generate(1, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
        result = w5.generate(1, 1, 2);
        assert_eq!(result, vec![0.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = Sum(
            Box::new(finite_const_gen(3.0, 1.0, 3.0)),
            Box::new(finite_const_gen(2.0, 0.0, 0.0)),
        );
        let result = w6.generate(1, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
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
