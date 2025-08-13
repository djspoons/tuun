use std::f32::consts::PI;
use std::fmt::Debug;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use fastrand;

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
pub enum Waveform {
    /*
     * Const produces a stream of samples where each sample is the same constant value.
     */
    Const(f32),
    /*
     * Time generates a stream based on the elapsed time from the start of the waveform, in seconds.
     */
    Time,
    /*
     * Noise generates random samples.
     */
    Noise,
    /*
     * Fixed generates the same, finite, sequence of samples.
     */
    Fixed(Vec<f32>),
    /*
     * Fin generates a finite waveform that lasts for the given duration, truncating
     * the underlying waveform.
     */
    Fin {
        duration: Duration,
        waveform: Box<Waveform>,
    },
    /*
     * Seq sets the offset to the given value (ignoring offset of the underlying waveform).
     */
    Seq {
        duration: Duration,
        waveform: Box<Waveform>,
    },
    /*
     * Sin computes the sine of each sample in the given waveform.
     */
    Sin(Box<Waveform>),
    /*
     * Convolution computes a new sample for each sample in the waveform by computing the sum of the products of that sample and the kernel.
     */
    Convolution {
        waveform: Box<Waveform>,
        kernel: Box<Waveform>,
    },
    Sum(Box<Waveform>, Box<Waveform>),
    DotProduct(Box<Waveform>, Box<Waveform>),
    /*
     * Res generates a repeating waveform that restarts the given waveform whenever the trigger
     * waveform flips from negative values to positive values. Its length and offset are determined
     * by the trigger waveform.
     */
    Res {
        trigger: Box<Waveform>,
        waveform: Box<Waveform>,
    },
    /*
     * Alt generates a waveform by alternating between two waveforms based on the sign of
     * the trigger waveform.
     */
    Alt {
        trigger: Box<Waveform>,
        positive_waveform: Box<Waveform>,
        negative_waveform: Box<Waveform>,
    },
    /*
     * Dial generates samples from an interactive "dial" input.
     */
    Dial(Dial),
    /*
     * Marked waveforms don't generate any samples, but are used to signal that a certain event will
     * occur or has occurred. Each status update will include a list of marked waveforms, along with
     * their start times and durations.
     */
    Marked {
        id: u32,
        waveform: Box<Waveform>,
    },
}

/*
 * Generator converts waveforms into sequences of samples.
 */
struct Generator {
    sample_frequency: i32,

    dial_values: std::collections::HashMap<Dial, f32>,
}

// TODO add metrics for waveform expr depth and total ops

impl Generator {
    fn new(sample_frequency: i32) -> Self {
        Generator {
            sample_frequency,
            dial_values: std::collections::HashMap::new(),
        }
    }

    // Generate a vector of samples up to `desired` length. `position` indicates where
    // the beginning of the result is relative to the start of the waveform. If fewer than
    // 'desired' samples are generated, that indicates that this waveform has finished (and
    // generate won't be called on it again).
    fn generate(&self, waveform: &Waveform, position: usize, desired: usize) -> Vec<f32> {
        match waveform {
            &Waveform::Const(value) => {
                return vec![value; desired];
            }
            Waveform::Time => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (i + position) as f32 / self.sample_frequency as f32;
                }
                return out;
            }
            Waveform::Noise => {
                let mut out = vec![0.0; desired];
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                return out;
            }
            Waveform::Fixed(samples) => {
                return samples.clone();
            }
            Waveform::Fin {
                waveform: inner_waveform,
                ..
            } => {
                let length = match self.length(waveform) {
                    Length::Finite(length) => length,
                    Length::Infinite => panic!("Finite waveform expected, got infinite"),
                };
                if position >= length {
                    return Vec::new(); // No samples to generate
                }
                return self.generate(inner_waveform, position, desired.min(length - position));
            }
            Waveform::Seq { waveform, .. } => {
                return self.generate(waveform, position, desired);
            }
            Waveform::Sin(waveform) => {
                let mut out = self.generate(waveform, position, desired);
                for f in out.iter_mut() {
                    *f = (2.0 * PI * *f).sin();
                }
                return out;
            }
            Waveform::Convolution { waveform, kernel } => {
                let kernel_length = match self.length(&kernel) {
                    Length::Finite(length) => length,
                    Length::Infinite => {
                        println!("Infinite kernel length, skipping generation");
                        return Vec::new();
                    }
                };
                let desired = match self.length(&waveform) {
                    Length::Finite(length) => {
                        if length + kernel_length / 2 > position {
                            desired.min(length + kernel_length / 2 - position)
                        } else {
                            return Vec::new();
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
                let waveform_out =
                    self.generate(waveform, position - position_diff, waveform_desired);
                let kernel_out = self.generate(kernel, 0, kernel_length);
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
                return out;
            }
            Waveform::Sum(a, b) => {
                return self.generate_binary_op(|x, y| x + y, a, b, position, desired);
            }
            Waveform::DotProduct(a, b) => {
                // Like sum, but we need to make sure we generate a length based on
                // the shorter waveform.
                let new_desired = match self.length(waveform) {
                    Length::Finite(length) => {
                        if length > position {
                            desired.min(length - position)
                        } else {
                            return Vec::new(); // No samples to generate
                        }
                    }
                    Length::Infinite => desired,
                };
                return self.generate_binary_op(|x, y| x * y, a, b, position, new_desired);
            }
            Waveform::Res { trigger, waveform } => {
                // TODO think about all of these unwrap_ors
                // TODO generate the trigger in blocks?
                // Maybe cache the last trigger position and signum and use it if position doesn't change?

                // First go back and find the most recent trigger before position.
                let mut last_trigger_position = position;
                let trigger_out = self.generate(trigger, position, 1);
                let mut last_signum = trigger_out.get(0).unwrap_or(&0.0).signum();
                while last_trigger_position > 0 {
                    let trigger_out = self.generate(trigger, last_trigger_position - 1, 1);
                    let new_signum = trigger_out.get(0).unwrap_or(&0.0).signum();
                    if last_signum >= 0.0 && new_signum < 0.0 {
                        break;
                    }
                    last_signum = new_signum;
                    last_trigger_position -= 1;
                }
                let mut inner_position = position - last_trigger_position;
                let mut generated = 0;
                let mut out = Vec::new();

                let trigger_out = self.generate(trigger, position, desired);

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

                    let mut tmp = self.generate(waveform, inner_position, inner_desired);
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
                return out;
            }
            Waveform::Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let mut out = self.generate(trigger, position, desired);
                let mut positive_out = self.generate(positive_waveform, position, desired);
                positive_out.resize(out.len(), 0.0);
                let mut negative_out = self.generate(negative_waveform, position, desired);
                negative_out.resize(out.len(), 0.0);
                for (i, x) in out.iter_mut().enumerate() {
                    if x.signum() >= 0.0 {
                        *x = positive_out[i];
                    } else {
                        *x = negative_out[i];
                    }
                }
                return out;
            }
            Waveform::Dial(dial) => {
                let value = self.dial_values.get(&dial).cloned().unwrap_or(0.0);
                return vec![value; desired];
            }
            Waveform::Marked { waveform, .. } => {
                return self.generate(waveform, position, desired);
            }
        }
    }

    // Returns the length of the waveform in samples.
    fn length(&self, waveform: &Waveform) -> Length {
        match waveform {
            Waveform::Const { .. } => Length::Infinite,
            Waveform::Time => Length::Infinite,
            Waveform::Noise => Length::Infinite,
            Waveform::Fixed(samples) => Length::Finite(samples.len()),
            Waveform::Fin { duration, .. } => {
                ((duration.as_secs_f32() * self.sample_frequency as f32) as usize).into()
            }
            Waveform::Seq { waveform, .. } => self.length(waveform),
            Waveform::Sin(waveform) => self.length(waveform),
            Waveform::Convolution { waveform, kernel } => {
                self.length(waveform) + self.length(kernel) / 2.into()
            }
            Waveform::Sum(a, b) => {
                let length = Length::Finite(self.offset(a)) + self.length(b);
                self.length(a).max(length)
            }
            Waveform::DotProduct(a, b) => {
                let length = Length::Finite(self.offset(a)) + self.length(b);
                self.length(a).min(length)
            }
            Waveform::Res { trigger, .. } | Waveform::Alt { trigger, .. } => self.length(trigger),
            Waveform::Dial { .. } => Length::Infinite,
            Waveform::Marked { waveform, .. } => self.length(waveform),
        }
    }

    fn offset(&self, waveform: &Waveform) -> usize {
        match waveform {
            Waveform::Const { .. } => 0,
            Waveform::Time => 0,
            Waveform::Noise => 0,
            Waveform::Fixed(_) => 0,
            Waveform::Fin { waveform, .. } => self.offset(waveform),
            Waveform::Seq { duration, .. } => {
                (duration.as_secs_f32() * self.sample_frequency as f32) as usize
            }
            Waveform::Sin(waveform) => self.offset(waveform),
            Waveform::Convolution { waveform, .. } => self.offset(waveform),
            Waveform::Sum(a, b) | Waveform::DotProduct(a, b) => self.offset(a) + self.offset(b),
            Waveform::Res { trigger, .. } | Waveform::Alt { trigger, .. } => self.offset(trigger),
            Waveform::Dial { .. } => 0,
            Waveform::Marked { waveform, .. } => self.offset(waveform),
        }
    }

    // Generate a binary operation on two waveforms, up to 'desired' samples starting at 'position'
    // relative to the start of the first waveform. The second waveform is offset by the
    // offset of the first waveform. The `op` function is applied to each pair of samples.
    fn generate_binary_op(
        &self,
        op: fn(f32, f32) -> f32,
        a: &Waveform,
        b: &Waveform,
        position: usize,
        desired: usize,
    ) -> Vec<f32> {
        let offset = self.offset(&a);
        let mut left = self.generate(a, position, desired);

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
                let right = self.generate(b, 0, desired - (offset - position));
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
                let right = self.generate(b, position - offset, desired);
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
}

pub enum Command<I> {
    Play {
        // A unique id for this waveform
        id: I,
        waveform: Waveform,
        // When the waveform should start playing; if in the past, then play immediately
        start: Instant,
        // If set, play this waveform in a loop
        repeat_every: Option<Duration>,
    },
    Stop {
        // The id of the waveform to stop
        id: I,
    },
    RemovePending {
        // The id of the waveform to remove
        id: I,
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
pub struct Mark<I> {
    pub waveform_id: I,
    pub mark_id: u32,
    pub start: Instant,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct Status<I>
where
    I: Clone + Send,
{
    pub buffer_start: Instant,
    // Marks for each active waveform as well as any pending waveforms
    pub marks: Vec<Mark<I>>,
    // Some status updates will include the current buffer
    pub buffer: Option<Vec<f32>>,
    // The current tracker load, the ratio of sample frequency to samples generated per second
    pub tracker_load: Option<f32>,
}

#[derive(Debug, Clone)]
struct ActiveWaveform<I>
where
    I: Clone,
{
    pub id: I,
    pub waveform: Waveform,
    pub marks: Vec<Mark<I>>,
    pub position: usize,
}

#[derive(Debug, Clone)]
struct PendingWaveform<I> {
    pub id: I,
    pub waveform: Waveform,
    pub start: Instant,
    pub repeat_every: Option<Duration>,
    pub marks: Vec<Mark<I>>,
}

pub struct Tracker<I>
where
    I: Clone + Send,
{
    sample_frequency: i32,
    command_receiver: Receiver<Command<I>>,
    status_sender: Sender<Status<I>>,

    // Internal state
    generator: Generator,
    active_waveforms: Vec<ActiveWaveform<I>>,
    pending_waveforms: Vec<PendingWaveform<I>>, // sorted by start time
    send_current_buffer: bool,
}

impl<I> Tracker<I>
where
    I: Clone + Send,
{
    pub fn new(
        sample_frequency: i32,
        command_receiver: Receiver<Command<I>>,
        status_sender: Sender<Status<I>>,
    ) -> Tracker<I> {
        return Tracker {
            sample_frequency,
            command_receiver,
            status_sender,

            generator: Generator::new(sample_frequency),

            active_waveforms: Vec::new(),
            pending_waveforms: Vec::new(),
            send_current_buffer: false,
        };
    }

    fn process_marked(
        &self,
        waveform_id: &I,
        start: Instant,
        waveform: &Waveform,
        out: &mut Vec<Mark<I>>,
    ) {
        use Waveform::{
            Alt, Const, Convolution, Dial, DotProduct, Fin, Fixed, Marked, Noise, Res, Seq, Sin,
            Sum, Time,
        };
        match waveform {
            Const(_) | Time | Noise | Fixed(_) | Dial { .. } => {
                return;
            }
            Fin { waveform, .. }
            | Seq { waveform, .. }
            | Sin(waveform)
            | Convolution { waveform, .. }
            | Res {
                trigger: waveform, ..
            }
            | Alt {
                trigger: waveform, ..
            } => {
                self.process_marked(waveform_id, start, &*waveform, out);
            }
            Sum(a, b) | DotProduct(a, b) => {
                self.process_marked(waveform_id, start, &*a, out);
                let offset = self.generator.offset(&*a);
                let start =
                    start + Duration::from_secs_f32(offset as f32 / self.sample_frequency as f32);
                self.process_marked(waveform_id, start, &*b, out);
            }
            Marked { waveform, id } => {
                let length = match self.generator.length(waveform) {
                    Length::Finite(length) => length,
                    Length::Infinite => usize::MAX, // Ehh, should we change the type of marks?
                };
                out.push(Mark {
                    waveform_id: waveform_id.clone(),
                    mark_id: *id,
                    start,
                    duration: Duration::from_secs_f32(length as f32 / self.sample_frequency as f32),
                });
                self.process_marked(waveform_id, start, &*waveform, out);
            }
        }
    }
}

impl<'a, I> AudioCallback for Tracker<I>
where
    I: Clone + PartialEq + Send + Debug,
{
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        // Check to see if we have any new commands
        self.empty_command_queue();

        // Assume that the callback is called with far enough in advance of when the samples are
        // needed that we can use time equal to the length of the buffer. If that's true, then
        // the moment corresponding to the start of the buffer is the current time plus the length
        // of the buffer.
        let buffer_start = Instant::now()
            + Duration::from_secs_f32(out.len() as f32 / self.sample_frequency as f32);
        let mut status_to_send = Status {
            buffer_start,
            marks: Vec::new(),
            tracker_load: None,
            buffer: None,
        };

        // Now generate!
        let generate_start = Instant::now();
        let (_, finished) = self.generate(buffer_start, out);
        status_to_send.tracker_load = Some(
            self.sample_frequency as f32
                / (out.len() as f32 / generate_start.elapsed().as_secs_f32()),
        );

        // Copy the marks from finished waveforms into the status
        for active in finished {
            status_to_send.marks.extend_from_slice(&active.marks);
        }
        // Copy the marks from active and pending waveforms into the status
        for active in &self.active_waveforms {
            status_to_send.marks.extend_from_slice(&active.marks);
        }
        for pending in &self.pending_waveforms {
            status_to_send.marks.extend_from_slice(&pending.marks);
        }

        if self.send_current_buffer {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            status_to_send.buffer = Some(copy);
            self.send_current_buffer = false;
        }

        self.status_sender.send(status_to_send).unwrap();
    }
}

impl<I> Tracker<I>
where
    I: Clone + PartialEq + Debug + Send,
{
    // buffer_start is the time corresponding to the beginning of the current buffer
    fn process_command(&mut self, command: Command<I>) {
        match command {
            Command::Play {
                id,
                waveform,
                start,
                repeat_every,
            } => {
                if let Some(duration) = repeat_every {
                    println!(
                        "Received command to play waveform {:?} at {:?} and every {:?}: {:?}",
                        id, start, duration, waveform
                    );
                } else {
                    println!(
                        "Received command to play waveform {:?} at {:?}: {:?}",
                        id, start, waveform
                    );
                }
                let mut marks = Vec::new();
                self.process_marked(&id, start, &waveform, &mut marks);
                self.pending_waveforms.push(PendingWaveform {
                    id,
                    waveform,
                    start,
                    repeat_every,
                    marks,
                });
                self.pending_waveforms.sort_by_key(|w| w.start);
            }
            Command::Stop { id } => {
                println!("Received command to stop waveform {:?}", id);
                self.active_waveforms.retain(|w| w.id != id);
            }
            Command::RemovePending { id } => {
                println!("Received command to remove pending waveform {:?}", id);
                self.pending_waveforms.retain(|w| w.id != id);
            }
            Command::SendCurrentBuffer => {
                self.send_current_buffer = true;
            }
            Command::TurnDial { dial, delta } => {
                //turn_dial_count += 1;
                // TODO accumulate the changes and apply them over the duration of the buffer
                self.generator
                    .dial_values
                    .entry(dial)
                    .and_modify(|v| *v += delta)
                    .or_insert(delta);
            }
        }
    }

    fn empty_command_queue(&mut self) {
        //println!("Dial state before processing commands: {:?}", self.generator.dial_values);
        //let mut turn_dial_count = 0;
        loop {
            match self.command_receiver.try_recv() {
                Ok(command) => self.process_command(command),
                Err(TryRecvError::Empty) => break,
                Err(e) => println!("Error receiving command: {:?}", e),
            }
        }
        //println!("Dial state after processing commands: {:?} ({} turns)", self.generator.dial_values, turn_dial_count);
    }

    // Generate from pending waveforms and active waveforms, filling the out buffer.
    // Returns how many samples were generated, or None if the no samples were generated
    // along with the set of active waveforms that finished generating
    fn generate(
        &mut self,
        buffer_start: Instant,
        out: &mut [f32],
    ) -> (Option<usize>, Vec<ActiveWaveform<I>>) {
        // We'll generate in segments based on the set of active waveforms at a given time.
        let mut segment_start = buffer_start;
        let mut segment_length = out.len();
        let mut finished = Vec::new();

        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut filled = 0; // How much of the out buffer we've filled so far
        let mut high_water_mark = 0; // How much that's filled by a waveform (not padded)
        while filled < out.len() {
            // Check to see if any pending waveform starts at or before segment_start. If so, promote
            // them active waveforms.
            while !self.pending_waveforms.is_empty() {
                if self.pending_waveforms[0].start <= segment_start {
                    let mut pending = self.pending_waveforms.remove(0);
                    println!(
                        "Activating waveform {:?} at time {:?}",
                        pending.id, segment_start
                    );
                    let mut marks = pending.marks;
                    if pending.start < segment_start {
                        // If the pending waveform starts before the segment start, then we need to
                        // adjust the marks to account for the segment start.
                        for mark in &mut marks {
                            mark.start += segment_start - pending.start;
                        }
                    }
                    self.active_waveforms.push(ActiveWaveform {
                        id: pending.id.clone(),
                        waveform: pending.waveform.clone(),
                        marks,
                        position: 0,
                    });
                    // Check to see if this waveform should repeat
                    if let Some(duration) = pending.repeat_every {
                        println!(
                            "Scheduling waveform {:?} to repeat after {:?}",
                            pending.id, duration
                        );
                        pending.start = segment_start + duration;
                        pending.marks = Vec::new();
                        self.process_marked(
                            &pending.id,
                            pending.start,
                            &pending.waveform,
                            &mut pending.marks,
                        );
                        self.pending_waveforms.push(pending);
                        self.pending_waveforms.sort_by_key(|w| w.start);
                    }
                } else {
                    // Set the length of the current segment to the start of the next pending waveform
                    // We take the ceiling here to make sure that we don't create a segment that is shorter
                    // than the duration of a single sample.
                    segment_length = segment_length.min(
                        ((self.pending_waveforms[0].start - segment_start).as_secs_f32()
                            * self.sample_frequency as f32)
                            .ceil() as usize,
                    );
                    break;
                }
            }

            // Finally, walk through the waveforms and generate samples up to the next start. If there
            // are no active waveforms, then just updated filled and continue.
            if self.active_waveforms.len() == 0 {
                filled += segment_length;
                segment_start +=
                    Duration::from_secs_f32(segment_length as f32 / self.sample_frequency as f32);
                segment_length = out.len() - filled;
                // Don't change high_water_mark
                continue;
            }

            let mut i = 0;
            while i < self.active_waveforms.len() {
                let active = &mut self.active_waveforms[i];
                let tmp =
                    self.generator
                        .generate(&active.waveform, active.position, segment_length);
                if tmp.len() > segment_length {
                    panic!(
                        "Generated more samples than desired: {} > {} for waveform id {:?} at position {}: {:?}", 
                        tmp.len(), segment_length, active.id, active.position, active.waveform);
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
                if tmp.len() < segment_length {
                    // If we didn't generate enough samples, then remove this waveform from the active list
                    println!(
                        "Removing waveform {:?} at position {}",
                        active.id, active.position
                    );
                    let active = self.active_waveforms.remove(i);
                    finished.push(active);
                } else {
                    active.position += segment_length;
                    i += 1;
                }
            }
            filled += segment_length;
            segment_start +=
                Duration::from_secs_f32(segment_length as f32 / self.sample_frequency as f32);
            segment_length = out.len() - filled;
            // Only set high_water_mark if there was at least one active waveform
            high_water_mark = filled;
        }
        if high_water_mark == 0 {
            (None, finished)
        } else {
            (Some(high_water_mark), finished)
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
            let generated = self.generate(Instant::now(), &mut out);
            // TODO double check to see if some padding is happening here
            match generated {
                (None, _) => break, // No more samples to generate
                (Some(n), _) => {
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
    use Waveform::{Const, Convolution, DotProduct, Fin, Res, Seq, Sin, Sum, Time};

    fn finite_const_gen(value: f32, fin_duration: u64, seq_duration: u64) -> Waveform {
        return Seq {
            duration: Duration::from_secs(seq_duration),
            waveform: Box::new(Fin {
                duration: Duration::from_secs(fin_duration),
                waveform: Box::new(Const(value)),
            }),
        };
    }

    fn run_tests(waveform: &Waveform, desired: Vec<f32>) {
        let generator = Generator::new(1);
        for size in [1, 2, 4, 8] {
            //println!("Running tests for waveform {:?} with size {}", waveform, size);
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = generator.generate(waveform, n * size, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_time() {
        let w1 = Waveform::Time;
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let generator = Generator::new(1);
        let result = generator.generate(&w1, 0, 8);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_res() {
        let w1 = Res {
            trigger: Box::new(Sin(Box::new(DotProduct(
                Box::new(Const(0.25)),
                Box::new(Time),
            )))),
            waveform: Box::new(Time),
        };
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w2 = Res {
            trigger: Box::new(Fin {
                duration: Duration::from_secs(6),
                waveform: Box::new(Sin(Box::new(DotProduct(
                    Box::new(Const(0.25)),
                    Box::new(Time),
                )))),
            }),
            waveform: Box::new(Time),
        };
        let generator = Generator::new(1);
        assert_eq!(generator.length(&w2), Length::Finite(6));
        run_tests(&w2, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0]);

        let w3 = Res {
            trigger: Box::new(Sin(Box::new(DotProduct(
                Box::new(Const(0.25)),
                Box::new(Time),
            )))),
            waveform: Box::new(Waveform::Fin {
                duration: Duration::from_secs(3),
                waveform: Box::new(Waveform::Time),
            }),
        };
        run_tests(&w3, vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_sum() {
        let generator = Generator::new(1);
        let w1 = Sum(
            Box::new(finite_const_gen(1.0, 5, 2)),
            Box::new(finite_const_gen(1.0, 5, 2)),
        );
        assert_eq!(generator.offset(&w1), 4);
        assert_eq!(generator.length(&w1), Length::Finite(7));
        run_tests(&w1, vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0]);

        let w2 = Fin {
            duration: Duration::from_secs(8),
            waveform: Box::new(Sum(
                Box::new(Seq {
                    duration: Duration::from_secs(0),
                    waveform: Box::new(Const(1.0)),
                }),
                Box::new(Sum(
                    Box::new(Seq {
                        duration: Duration::from_secs(0),
                        waveform: Box::new(Const(2.0)),
                    }),
                    Box::new(Fin {
                        duration: Duration::from_secs(0),
                        waveform: Box::new(Const(0.0)),
                    }),
                )),
            )),
        };
        run_tests(&w2, vec![3.0; 8]);

        let w5 = Sum(
            Box::new(finite_const_gen(3.0, 1, 3)),
            Box::new(finite_const_gen(2.0, 2, 2)),
        );
        run_tests(&w5, vec![3.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0]);

        // Test a case to make sure that the sum generates enough samples, even when
        // the left-hand side is shorter and the right hasn't started yet.
        let result = generator.generate(&w5, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
        let result = generator.generate(&w5, 1, 2);
        assert_eq!(result, vec![0.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = Sum(
            Box::new(finite_const_gen(3.0, 1, 3)),
            Box::new(finite_const_gen(2.0, 0, 0)),
        );
        let result = generator.generate(&w6, 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let generator = Generator::new(1);
        let w1 = DotProduct(
            Box::new(finite_const_gen(3.0, 8, 2)),
            Box::new(finite_const_gen(2.0, 5, 2)),
        );
        assert_eq!(generator.offset(&w1), 4);
        assert_eq!(generator.length(&w1), Length::Finite(7));
        run_tests(&w1, vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0]);

        let w2 = DotProduct(
            Box::new(finite_const_gen(3.0, 5, 2)),
            Box::new(finite_const_gen(2.0, 5, 2)),
        );
        run_tests(&w2, vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]);

        let w3 = Fin {
            duration: Duration::from_secs(8),
            waveform: Box::new(DotProduct(Box::new(Const(3.0)), Box::new(Const(2.0)))),
        };
        run_tests(&w3, vec![6.0; 8]);

        let w4 = DotProduct(
            Box::new(Seq {
                duration: Duration::from_secs(1),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_gen(2.0, 5, 5)),
        );
        run_tests(&w4, vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_convolution() {
        let w1 = Convolution {
            waveform: Box::new(Time),
            kernel: Box::new(finite_const_gen(2.0, 3, 3)),
        };
        run_tests(&w1, vec![2.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]);

        let w2 = Convolution {
            waveform: Box::new(Fin {
                duration: Duration::from_secs(5),
                waveform: Box::new(Time),
            }),
            kernel: Box::new(finite_const_gen(2.0, 3, 3)),
        };
        let generator = Generator::new(1);
        assert_eq!(generator.length(&w2), Length::Finite(6));
        run_tests(&w2, vec![2.0, 6.0, 12.0, 18.0, 14.0, 8.0, 0.0, 0.0]);

        let w3 = Convolution {
            waveform: Box::new(Fin {
                duration: Duration::from_secs(3),
                waveform: Box::new(Time),
            }),
            kernel: Box::new(finite_const_gen(2.0, 5, 5)),
        };
        let generator = Generator::new(1);
        assert_eq!(generator.length(&w3), Length::Finite(5));
        run_tests(&w3, vec![6.0, 6.0, 6.0, 6.0, 4.0, 0.0, 0.0, 0.0]);

        let w4 = Convolution {
            waveform: Box::new(Res {
                trigger: Box::new(Sin(Box::new(DotProduct(
                    Box::new(Const(1.0 / 3.0)),
                    Box::new(Time),
                )))),
                waveform: Box::new(Time),
            }),
            kernel: Box::new(finite_const_gen(2.0, 5, 5)),
        };
        run_tests(&w4, vec![6.0, 6.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0]);

        let w5 = Convolution {
            waveform: Box::new(Const(1.0)),
            kernel: Box::new(finite_const_gen(0.2, 5, 5)),
        };
        run_tests(&w5, vec![0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
