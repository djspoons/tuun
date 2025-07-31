use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

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

impl std::ops::Div for Length {
    type Output = Length;

    fn div(self, other: Length) -> Length {
        match (self, other) {
            (Length::Finite(a), Length::Finite(b)) => Length::Finite(a / b),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dial {
    X,
    Y,
}

#[derive(Debug, Clone)]
pub enum Waveform<S> {
    /*
     * SineWave is a sinusoidal wave generator with the given frequency.
     */
    SineWave {
        frequency: Box<Waveform<S>>,
    },
    Const(f32),
    /*
     * Time generates a signal based on the current time from the start of this waveform.
     */
    Time,
    /*
     * Dial generates a continuous control signal from a "dial" input.
     */
    Dial(Dial),
    /*
     * Fin generates a finite waveform that lasts for the given duration in beats, truncating
     * the underlying waveform.
     */
    Fin {
        duration: f32, // duration in beats
        waveform: Box<Waveform<S>>,
    },
    /*
     * Seq sets the offset to the given value (ignoring offset of the underlying waveform).
     */
    Seq {
        duration: f32, // duration in beats
        waveform: Box<Waveform<S>>,
    },
    /*
     * Rep generates a repeating waveform that loops the given waveform whenever the trigger waveform flips from negative values to positive values. Its length and offset are determined by the trigger waveform.
     */
    Rep {
        trigger: Box<Waveform<S>>,
        waveform: Box<Waveform<S>>,
        state: S,
    },
    Sum(Box<Waveform<S>>, Box<Waveform<S>>),
    DotProduct(Box<Waveform<S>>, Box<Waveform<S>>),
    /*
     * Convolution computes a new sample for each sample in the waveform by computing the sum of the products of that sample and the kernel.
     */
    Convolution {
        waveform: Box<Waveform<S>>,
        kernel: Box<Waveform<S>>,
    },
}

/*
 * Generator converts waveforms into sequences of samples.
 */
struct Generator {
    sample_frequency: i32,
    beats_per_minute: u32,

    dial_values: std::collections::HashMap<Dial, f32>,
}

// TODO add metrics for waveform expr depth and total ops

impl Generator {
    fn new(sample_frequency: i32, beats_per_minute: u32) -> Self {
        Generator {
            sample_frequency,
            beats_per_minute,
            dial_values: std::collections::HashMap::new(),
        }
    }

    // Generate a vector of samples up to `desired` length. `position` indicates where
    // the beginning of the result is relative to the start of the waveform. If fewer than
    // 'desired' samples are generated, that indicates that this waveform has finished (and
    // generate won't be called on it again).
    fn generate(
        &self,
        waveform: Waveform<(f32, usize)>,
        position: usize,
        desired: usize,
    ) -> (Waveform<(f32, usize)>, Vec<f32>) {
        match waveform {
            Waveform::SineWave { frequency } => {
                let (frequency, mut out) = self.generate(*frequency, position, desired);
                for (i, f) in out.iter_mut().enumerate() {
                    let t_secs = (i + position) as f32 / self.sample_frequency as f32;
                    *f = (2.0 * PI * *f * t_secs).sin();
                }
                return (
                    Waveform::SineWave {
                        frequency: Box::new(frequency),
                    },
                    out,
                );
            }
            Waveform::Const(value) => {
                return (waveform, vec![value; desired]);
            }
            Waveform::Time => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (i + position) as f32 / self.sample_frequency as f32;
                }
                return (waveform, out);
            }
            Waveform::Dial(dial) => {
                let value = self.dial_values.get(&dial).cloned().unwrap_or(0.0);
                return (waveform, vec![value; desired]);
            }
            Waveform::Fin {
                duration,
                waveform: inner_waveform,
            } => {
                // TODO if generate took &Waveform then we could call length()
                let length = (duration
                    * samples_per_beat(self.sample_frequency, self.beats_per_minute) as f32)
                    as usize;
                if position >= length {
                    return (
                        Waveform::Fin {
                            duration,
                            waveform: inner_waveform,
                        },
                        Vec::new(),
                    ); // No samples to generate
                }
                let (inner_waveform, out) =
                    self.generate(*inner_waveform, position, desired.min(length - position));
                return (
                    Waveform::Fin {
                        duration,
                        waveform: Box::new(inner_waveform),
                    },
                    out,
                );
            }
            Waveform::Rep {
                mut trigger,
                mut waveform,
                state: (mut _last_signum, mut _inner_position),
            } => {
                // TODO think about all of these unwrap_ors
                // TODO generate the trigger in blocks?
                // TODO cache the last trigger position and signum and use it if position doesn't change
                // First go back and find the most recent trigger before position.
                let mut last_trigger_position = position;
                let (tmp_trigger, out) = self.generate(*trigger, position, 1);
                trigger = Box::new(tmp_trigger);
                let mut last_signum = out.get(0).unwrap_or(&0.0).signum();
                while last_trigger_position > 0 {
                    let (tmp_trigger, out) = self.generate(*trigger, last_trigger_position - 1, 1);
                    trigger = Box::new(tmp_trigger);
                    let new_signum = out.get(0).unwrap_or(&0.0).signum();
                    if last_signum >= 0.0 && new_signum < 0.0 {
                        break;
                    }
                    last_signum = new_signum;
                    last_trigger_position -= 1;
                }
                let mut inner_position = position - last_trigger_position;
                let mut generated = 0;
                let mut out = Vec::new();

                let (trigger, trigger_out) = self.generate(*trigger, position, desired);

                while generated < trigger_out.len() {
                    // Set to true if there a restart will be triggered before desired
                    let mut reset_inner_position = false;
                    let mut inner_desired = trigger_out.len() - generated;

                    for (i, &x) in trigger_out[generated..].iter().enumerate() {
                        if last_signum < 0.0 && x >= 0.0 {
                            inner_desired = i;
                            reset_inner_position = true;
                            last_signum = x.signum();
                            break;
                        } else if last_signum >= 0.0 && x < 0.0 {
                            last_signum = x.signum();
                        }
                    }

                    let (inner_waveform, mut tmp) =
                        self.generate(*waveform, inner_position, inner_desired);
                    waveform = Box::new(inner_waveform);
                    if tmp.len() < inner_desired {
                        tmp.resize(inner_desired, 0.0);
                    }
                    out.extend(tmp);
                    generated += inner_desired;
                    if reset_inner_position {
                        inner_position = 0;
                    } else {
                        inner_position += inner_desired;
                    }
                }
                return (
                    Waveform::Rep {
                        trigger: Box::new(trigger),
                        waveform,
                        state: (last_signum, inner_position),
                    },
                    out,
                );
            }
            Waveform::Seq {
                duration, waveform, ..
            } => {
                let (waveform, out) = self.generate(*waveform, position, desired);
                return (
                    Waveform::Seq {
                        duration,
                        waveform: Box::new(waveform),
                    },
                    out,
                );
            }
            Waveform::Sum(a, b) => {
                let (a, b, out) = self.generate_binary_op(|x, y| x + y, *a, *b, position, desired);
                return (Waveform::Sum(Box::new(a), Box::new(b)), out);
            }
            Waveform::DotProduct(a, b) => {
                // Like sum, but we need to make sure we generate a length based on
                // the shorter waveform.
                // TODO if generate took &Waveform then we wouldn't need to reconstruct here
                let new_desired = match self.length(&Waveform::DotProduct(a.clone(), b.clone())) {
                    Length::Finite(length) => {
                        if length > position {
                            desired.min(length - position)
                        } else {
                            return (Waveform::DotProduct(a, b), Vec::new()); // No samples to generate
                        }
                    }
                    Length::Infinite => desired,
                };
                let (a, b, out) =
                    self.generate_binary_op(|x, y| x * y, *a, *b, position, new_desired);
                return (Waveform::DotProduct(Box::new(a), Box::new(b)), out);
            }
            Waveform::Convolution { waveform, kernel } => {
                let kernel_length = match self.length(&kernel) {
                    Length::Finite(length) => length,
                    Length::Infinite => {
                        println!("Infinite kernel length, skipping generation");
                        return (Waveform::Convolution { waveform, kernel }, Vec::new());
                    }
                };
                let desired = match self.length(&waveform) {
                    Length::Finite(length) => {
                        if length + kernel_length / 2 > position {
                            desired.min(length + kernel_length / 2 - position)
                        } else {
                            return (Waveform::Convolution { waveform, kernel }, Vec::new());
                        }
                    }
                    Length::Infinite => desired,
                };

                // We want to generate additional samples on both ends to convolve with the kernel.
                // position_diff is the number of additional samples generated on the left side.
                let (position_diff, waveform_desired) = if position >= kernel_length / 2 {
                    (kernel_length / 2, desired + kernel_length - 1)
                } else {
                    let position_diff = position.min(kernel_length / 2 - position);
                    (position_diff, desired + position_diff + kernel_length / 2)
                };
                let (waveform, waveform_out) =
                    self.generate(*waveform, position - position_diff, waveform_desired);
                let (kernel, kernel_out) = self.generate(*kernel, 0, kernel_length);
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    for (j, &k) in kernel_out.iter().enumerate() {
                        if i + j + position_diff >= kernel_length / 2 {
                            let a = waveform_out
                                .get(i + j + position_diff - (kernel_length / 2))
                                .unwrap_or(&0.0);
                            *x += a * k;
                        }
                    }
                }
                return (
                    Waveform::Convolution {
                        waveform: Box::new(waveform),
                        kernel: Box::new(kernel),
                    },
                    out,
                );
            }
        }
    }

    fn length<S>(&self, waveform: &Waveform<S>) -> Length {
        match waveform {
            Waveform::Const { .. } => Length::Infinite,
            Waveform::Time => Length::Infinite,
            Waveform::Dial { .. } => Length::Infinite,
            Waveform::Fin { duration, .. } => ((duration
                * samples_per_beat(self.sample_frequency, self.beats_per_minute) as f32)
                as usize)
                .into(),
            Waveform::Rep { trigger, .. } => self.length(trigger),
            Waveform::Seq { waveform, .. } => self.length(waveform),
            Waveform::SineWave { frequency } => self.length(frequency),
            Waveform::Sum(a, b) => {
                let length = Length::Finite(self.offset(a)) + self.length(b);
                self.length(a).max(length)
            }
            Waveform::DotProduct(a, b) => {
                let length = Length::Finite(self.offset(a)) + self.length(b);
                self.length(a).min(length)
            }
            Waveform::Convolution { waveform, kernel } => {
                self.length(waveform) + self.length(kernel) / 2.into()
            }
        }
    }

    fn offset<S>(&self, waveform: &Waveform<S>) -> usize {
        match waveform {
            Waveform::Const { .. } => 0,
            Waveform::Time => 0,
            Waveform::Dial { .. } => 0,
            Waveform::Fin { waveform, .. } => self.offset(waveform),
            Waveform::Rep { trigger, .. } => self.offset(trigger),
            Waveform::Seq { duration, .. } => {
                (duration * samples_per_beat(self.sample_frequency, self.beats_per_minute) as f32)
                    as usize
            }
            Waveform::SineWave { frequency } => self.offset(frequency),
            Waveform::Sum(a, b) | Waveform::DotProduct(a, b) => self.offset(a) + self.offset(b),
            Waveform::Convolution { waveform, .. } => self.offset(waveform),
        }
    }

    // Generate a binary operation on two waveforms, up to 'desired' samples starting at 'position'
    // relative to the start of the first waveform. The second waveform is offset by the
    // offset of the first waveform. The `op` function is applied to each pair of samples.
    fn generate_binary_op(
        &self,
        op: fn(f32, f32) -> f32,
        a: Waveform<(f32, usize)>,
        mut b: Waveform<(f32, usize)>,
        position: usize,
        desired: usize,
    ) -> (Waveform<(f32, usize)>, Waveform<(f32, usize)>, Vec<f32>) {
        let offset = self.offset(&a);
        let (a, mut left) = self.generate(a, position, desired);

        if offset >= position + desired {
            // Make sure the left side is long enough so that we get another chance to
            // generate the right-hand side.
            left.resize(desired, 0.0);
        } else {
            // offset < position + desired
            // There is an overlap between the desired portion and the right waveform...
            //    1) ... and the right waveform starts after position
            // or 2) ... and the right waveform starts before position

            if position + left.len() < offset {
                // Either way, if the left side is shorter than the next offset, than extend it.
                left.resize(offset - position, 0.0);
            }

            if position < offset {
                // ... and the right waveform starts after position
                let (tmp_b, right) = self.generate(b, 0, desired - (offset - position));
                b = tmp_b;
                // Merge the overlapping portion
                for (i, x) in left[offset - position..].iter_mut().enumerate() {
                    if i >= right.len() {
                        break;
                    }
                    *x = op(*x, right[i]);
                }
                // If the left side is shorter than the right, than append.
                if right.len() + offset > left.len() + position {
                    left.extend_from_slice(&right[(left.len() + position - offset)..]);
                }
            } else {
                // ... and the right waveform starts before  position
                let (tmp_b, right) = self.generate(b, position - offset, desired);
                b = tmp_b;
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
        return (a, b, left);
    }
}

pub enum Command {
    PlayOnce {
        // A unique id for this waveform
        id: u32,
        waveform: Waveform<()>,
        // When the waveform should start playing, in beats from the beginning of playback;
        // if None, then play immediately.
        at_beat: Option<u64>,
    },
    SendCurrentBuffer,
    TurnDial {
        // The dial to set
        dial: Dial,
        // The amount to change it by
        delta: f32,
    },
}

#[derive(Debug, Clone)]
pub struct ActiveWaveform {
    pub id: u32,
    pub waveform: Waveform<(f32, usize)>,
    pub position: usize,
}

#[derive(Debug, Clone)]
pub struct PendingWaveform {
    pub id: u32,
    pub waveform: Waveform<()>,
    pub beat: Option<u64>, // The beat at which this waveform should be played
}

#[derive(Debug, Clone)]
pub struct Status {
    pub active_waveforms: Vec<ActiveWaveform>,
    pub pending_waveforms: Vec<PendingWaveform>,
    pub current_beat: u64,
    pub next_beat_start: Instant,
    pub tracker_load: Option<f32>, // ratio of sample frequency to samples generated per second
    pub buffer: Option<Vec<f32>>,
}

pub struct Tracker {
    sample_frequency: i32,
    beats_per_minute: u32,
    command_receiver: Receiver<Command>,
    status_sender: Sender<Status>,

    // Internal state
    generator: Generator,
    active_waveforms: Vec<ActiveWaveform>,
    pending_waveforms: Vec<PendingWaveform>,
    current_beat: u64,
    samples_to_next_beat: usize,
    send_current_buffer: bool,
}

impl Tracker {
    pub fn new(
        sample_frequency: i32,
        beats_per_minute: u32,
        command_receiver: Receiver<Command>,
        status_sender: Sender<Status>,
    ) -> Tracker {
        return Tracker {
            sample_frequency,
            beats_per_minute,
            command_receiver,
            status_sender,

            generator: Generator::new(sample_frequency, beats_per_minute),

            active_waveforms: Vec::new(),
            pending_waveforms: Vec::new(),
            current_beat: 1,
            samples_to_next_beat: samples_per_beat(sample_frequency, beats_per_minute),
            send_current_buffer: false,
        };
    }
}
fn samples_per_beat(sample_frequency: i32, beats_per_minute: u32) -> usize {
    // (seconds/minute) * 60/(beats/min) * (samples/sec)
    let seconds_per_beat = 60.0 / beats_per_minute as f32;
    (sample_frequency as f32 * seconds_per_beat) as usize
}

impl<'a> AudioCallback for Tracker {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        // Check to see if we have any new commands
        self.empty_command_queue();
        let mut status_to_send = Status {
            active_waveforms: self.active_waveforms.clone(),
            pending_waveforms: self.pending_waveforms.clone(),
            current_beat: self.current_beat,
            next_beat_start: Instant::now()
                + std::time::Duration::from_millis(
                    (self.samples_to_next_beat as f32 / self.sample_frequency as f32 * 1000.0)
                        as u64,
                ),
            tracker_load: None,
            buffer: None,
        };

        // Now generate!
        let generate_start = Instant::now();
        let _ = self.generate(out);
        status_to_send.tracker_load = Some(
            self.sample_frequency as f32
                / (out.len() as f32 / generate_start.elapsed().as_secs_f32()),
        );

        if self.send_current_buffer {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            status_to_send.buffer = Some(copy);
            self.send_current_buffer = false;
        }

        self.status_sender.send(status_to_send).unwrap();
    }
}

fn make_active_waveform(a: Waveform<()>) -> Waveform<(f32, usize)> {
    match a {
        Waveform::SineWave { frequency } => Waveform::SineWave {
            frequency: Box::new(make_active_waveform(*frequency)),
        },
        Waveform::Const(value) => Waveform::Const(value),
        Waveform::Time => Waveform::Time,
        Waveform::Dial(dial) => Waveform::Dial(dial),
        Waveform::Fin { duration, waveform } => Waveform::Fin {
            duration,
            waveform: Box::new(make_active_waveform(*waveform)),
        },
        Waveform::Seq { duration, waveform } => Waveform::Seq {
            duration,
            waveform: Box::new(make_active_waveform(*waveform)),
        },
        Waveform::Rep {
            trigger, waveform, ..
        } => Waveform::Rep {
            trigger: Box::new(make_active_waveform(*trigger)),
            waveform: Box::new(make_active_waveform(*waveform)),
            state: ((-1.0f32).signum(), 0),
        },
        Waveform::Sum(a, b) => Waveform::Sum(
            Box::new(make_active_waveform(*a)),
            Box::new(make_active_waveform(*b)),
        ),
        Waveform::DotProduct(a, b) => Waveform::DotProduct(
            Box::new(make_active_waveform(*a)),
            Box::new(make_active_waveform(*b)),
        ),
        Waveform::Convolution { waveform, kernel } => Waveform::Convolution {
            waveform: Box::new(make_active_waveform(*waveform)),
            kernel: Box::new(make_active_waveform(*kernel)),
        },
    }
}

impl Tracker {
    fn empty_command_queue(&mut self) {
        //println!("Dial state before processing commands: {:?}", self.generator.dial_values);
        //let mut turn_dial_count = 0;
        loop {
            match self.command_receiver.try_recv() {
                Ok(Command::PlayOnce {
                    id,
                    waveform,
                    at_beat,
                }) => {
                    println!(
                        "Received command to play once at beat {:?} with waveform {}: {:?}",
                        at_beat, id, waveform
                    );
                    match at_beat {
                        Some(beat) if beat < self.current_beat => {
                            println!(
                                "Ignoring command to play waveform {} at beat {:?} (current beat is {})",
                                id, at_beat, self.current_beat
                           );
                        }
                        None => {
                            // Play immediately
                            self.active_waveforms.push(ActiveWaveform {
                                id,
                                waveform: make_active_waveform(waveform),
                                position: 0,
                            });
                        }
                        _ => {
                            self.pending_waveforms.push(PendingWaveform {
                                id,
                                waveform,
                                beat: at_beat,
                            });
                        }
                    }
                }
                Ok(Command::SendCurrentBuffer) => {
                    self.send_current_buffer = true;
                }
                Ok(Command::TurnDial { dial, delta }) => {
                    //turn_dial_count += 1;
                    self.generator
                        .dial_values
                        .entry(dial)
                        .and_modify(|v| *v += delta)
                        .or_insert(delta);
                }
                Err(TryRecvError::Empty) => {
                    break;
                }
                Err(e) => {
                    println!("Error receiving command: {:?}", e);
                }
            }
        }
        //println!("Dial state after processing commands: {:?} ({} turns)", self.generator.dial_values, turn_dial_count);
    }

    // Generate from pending waveforms and active waveforms, filling the out buffer.
    // Returns how many samples were generated, or None if the no samples were generated.
    fn generate(&mut self, out: &mut [f32]) -> Option<usize> {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut filled = 0; // How much of the out buffer we've filled so far
        let mut generated = 0; // How many samples we actually generated (vs filled w/ 0s)
        while filled < out.len() {
            let mut desired: usize = out.len() - filled;
            if self.samples_to_next_beat == 0 {
                self.samples_to_next_beat =
                    samples_per_beat(self.sample_frequency, self.beats_per_minute);
                self.current_beat += 1;

                // If we are at the start of a beat, check to see if there any any pending waveforms
                // that can become active waveforms.
                let mut i = 0;
                while i < self.pending_waveforms.len() {
                    let pending = &self.pending_waveforms[i];
                    if pending.beat == Some(self.current_beat) {
                        // This waveform can become active
                        self.active_waveforms.push(ActiveWaveform {
                            id: pending.id,
                            waveform: make_active_waveform(pending.waveform.clone()),
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
                let (waveform, tmp) =
                    self.generator
                        .generate(active.waveform.clone(), active.position, desired);
                active.waveform = waveform;
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
                    for (j, &x) in tmp.iter().enumerate() {
                        out[filled + j] += x;
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
            generated += desired;
        }
        if generated == 0 {
            None
        } else {
            Some(generated)
        }
    }

    pub fn write_to_file(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use hound;

        self.empty_command_queue();

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(filename, spec)?;
        loop {
            let mut out = vec![0.0; 1024];
            let generated = self.generate(&mut out);
            // TODO double check to see if some padding is happening here
            match generated {
                None => break, // No more samples to generate
                Some(n) => {
                    for x in out[..n].iter() {
                        writer.write_sample(*x)?;
                    }
                }
            }
        }
        return writer.finalize().map_err(|e| e.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Waveform::{Const, Convolution, DotProduct, Fin, Rep, Seq, SineWave, Sum, Time};

    fn finite_const_gen<S>(value: f32, fin_duration: f32, seq_duration: f32) -> Waveform<S> {
        return Seq {
            duration: seq_duration,
            waveform: Box::new(Fin {
                duration: fin_duration,
                waveform: Box::new(Const(value)),
            }),
        };
    }

    fn run_tests(waveform: Waveform<(f32, usize)>, desired: Vec<f32>) -> Waveform<(f32, usize)> {
        let generator = Generator::new(1, 60);
        for size in [1, 2, 4, 8] {
            //println!("Running tests for waveform {:?} with size {}", waveform, size);
            let mut waveform = waveform.clone();
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (w, tmp) = generator.generate(waveform, n * size, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
                waveform = w;
            }
            assert_eq!(out, desired);
        }
        waveform
    }

    #[test]
    fn test_time() {
        let mut w1 = Waveform::Time;
        w1 = run_tests(w1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let generator = Generator::new(1, 90);
        let (_, result) = generator.generate(w1, 0, 8);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_rep() {
        let w1 = make_active_waveform(Rep {
            trigger: Box::new(SineWave {
                frequency: Box::new(Const(0.25)),
            }),
            waveform: Box::new(Time),
            state: (),
        });
        run_tests(w1, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w2 = make_active_waveform(Rep {
            trigger: Box::new(Fin {
                duration: 6.0,
                waveform: Box::new(SineWave {
                    frequency: Box::new(Const(0.25)),
                }),
            }),
            waveform: Box::new(Time),
            state: (),
        });
        let generator = Generator::new(1, 60);
        assert_eq!(generator.length(&w2), Length::Finite(6));
        run_tests(w2, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0]);

        let w3 = make_active_waveform(Rep {
            trigger: Box::new(Waveform::SineWave {
                frequency: Box::new(Waveform::Const(0.25)),
            }),
            waveform: Box::new(Waveform::Fin {
                duration: 3.0,
                waveform: Box::new(Waveform::Time),
            }),
            state: (),
        });
        run_tests(w3, vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_sum() {
        let generator = Generator::new(1, 60);
        let w1 = Sum(
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
        );
        assert_eq!(generator.offset(&w1), 4);
        assert_eq!(generator.length(&w1), Length::Finite(7));
        run_tests(w1, vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0]);

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
        run_tests(w2, vec![3.0; 8]);

        let mut w5 = Sum(
            Box::new(finite_const_gen(3.0, 1.0, 3.0)),
            Box::new(finite_const_gen(2.0, 2.0, 2.0)),
        );
        w5 = run_tests(w5, vec![3.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0]);

        // Test a case to make sure that the sum generates enough samples, even when
        // the left-hand side is shorter and the right hasn't started yet.
        let (w5, result) = generator.generate(w5, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
        let (_, result) = generator.generate(w5, 1, 2);
        assert_eq!(result, vec![0.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = Sum(
            Box::new(finite_const_gen(3.0, 1.0, 3.0)),
            Box::new(finite_const_gen(2.0, 0.0, 0.0)),
        );
        let (_, result) = generator.generate(w6, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let generator = Generator::new(1, 60);
        let w1 = DotProduct(
            Box::new(finite_const_gen(3.0, 8.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        assert_eq!(generator.offset(&w1), 4);
        assert_eq!(generator.length(&w1), Length::Finite(7));
        run_tests(w1, vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0]);

        let w2 = DotProduct(
            Box::new(finite_const_gen(3.0, 5.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        run_tests(w2, vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]);

        let w3 = Fin {
            duration: 8.0,
            waveform: Box::new(DotProduct(Box::new(Const(3.0)), Box::new(Const(2.0)))),
        };
        run_tests(w3, vec![6.0; 8]);

        let w4 = DotProduct(
            Box::new(Seq {
                duration: 1.0,
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_gen(2.0, 5.0, 5.0)),
        );
        run_tests(w4, vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_convolution() {
        let w1 = Convolution {
            waveform: Box::new(Time),
            kernel: Box::new(finite_const_gen(2.0, 3.0, 3.0)),
        };
        run_tests(w1, vec![2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]);

        let w2 = Convolution {
            waveform: Box::new(Fin {
                duration: 5.0,
                waveform: Box::new(Time),
            }),
            kernel: Box::new(finite_const_gen(2.0, 3.0, 3.0)),
        };
        let generator = Generator::new(1, 60);
        assert_eq!(generator.length(&w2), Length::Finite(6));
        run_tests(w2, vec![2.0, 6.0, 12.0, 18.0, 14.0, 8.0, 0.0, 0.0]);

        let w3 = Convolution {
            waveform: Box::new(Fin {
                duration: 3.0,
                waveform: Box::new(Time),
            }),
            kernel: Box::new(finite_const_gen(2.0, 5.0, 5.0)),
        };
        let generator = Generator::new(1, 60);
        assert_eq!(generator.length(&w3), Length::Finite(5));
        run_tests(w3, vec![6.0, 6.0, 6.0, 6.0, 4.0, 0.0, 0.0, 0.0]);

        let w4 = Convolution {
            waveform: Box::new(make_active_waveform(Rep {
                trigger: Box::new(SineWave {
                    frequency: Box::new(Const(1.0 / 3.0)),
                }),
                waveform: Box::new(Time),
                state: (),
            })),
            kernel: Box::new(finite_const_gen(2.0, 5.0, 5.0)),
        };
        run_tests(w4, vec![6.0, 6.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0]);
    }
}
