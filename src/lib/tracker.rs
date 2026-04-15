use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::io::BufWriter;
use std::path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

extern crate sdl2;
use sdl2::audio;

use crate::generator;
use crate::waveform;

pub enum Command<I, M> {
    Play {
        // A unique id for this waveform
        id: I,
        waveform: waveform::Waveform<M>,
        // When the waveform should start playing; if in the past or None, then play immediately
        start: Option<Instant>,
        // If set, play this waveform in a loop
        repeat_every: Option<Duration>,
    },
    // Immediately modify the waveform with the given id to replace the contents of any marked waveform
    // with the given mark_id with the new waveform.
    Modify {
        id: I,
        mark_id: M,
        waveform: waveform::Waveform<M>,
    },
    RemovePending {
        // The id of the waveform to remove
        id: I,
    },
    SendCurrentBuffer,
}

#[derive(Debug, Clone)]
pub struct Mark<I, M> {
    pub waveform_id: I,
    pub mark_id: M,
    pub start: Instant,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct Status<I, M>
where
    I: Clone + Send,
    M: Clone + Send,
{
    pub buffer_start: Instant,
    // Marks for each active waveform as well as any pending waveforms; a mark may appear more
    // than once if a given waveform is both active and pending
    pub marks: Vec<Mark<I, M>>,
    // Some status updates will include the current buffer
    pub buffer: Option<Vec<f32>>,
    // The current tracker load, the ratio of sample frequency to samples generated per second
    pub tracker_load: Option<f32>,
    // The number of samples allocated internally as part of generating one sample (on average)
    pub allocations_per_sample: Option<f32>,
}

impl<I, M> Status<I, M>
where
    I: Clone + Send + PartialEq,
    M: Clone + Send + PartialEq,
{
    pub fn has_pending_mark(&self, when: Instant, id: I, mark: M) -> bool {
        self.marks
            .iter()
            .any(|w| w.waveform_id == id && w.mark_id == mark && w.start > when)
    }

    pub fn has_active_mark(&self, when: Instant, id: I, mark: M) -> bool {
        self.marks
            .iter()
            .any(|w| w.waveform_id == id && w.mark_id == mark && w.start <= when)
    }
}

struct ActiveWaveform<I, M>
where
    I: Clone,
    M: Clone,
{
    id: I,
    waveform: generator::Waveform<M>,
    start: Instant,
    marks: Vec<Mark<I, M>>,
    // Open files used by Captured waveforms
    capture_state: HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
}

#[derive(Debug, Clone)]
struct PendingWaveform<I, M>
where
    M: Clone,
{
    id: I,
    // TODO it seems like pending shouldn't have state, but because of the way
    // we currently process marks, we do need state. Remove someday?
    waveform: generator::Waveform<M>,
    start: Instant,
    repeat_every: Option<Duration>,
    marks: Vec<Mark<I, M>>,
}

pub struct Tracker<'a, I, M>
where
    I: Clone + Send,
    M: Clone + Send + fmt::Display,
{
    generator: generator::Generator<'a>,
    sample_rate: i32,
    captured_output_dir: path::PathBuf,
    captured_date_format: String,
    command_receiver: mpsc::Receiver<Command<I, M>>,
    status_sender: mpsc::Sender<Status<I, M>>,

    // Persistent generation state
    active_waveforms: Vec<ActiveWaveform<I, M>>,
    pending_waveforms: Vec<PendingWaveform<I, M>>, // sorted by start time
    // Command state
    send_current_buffer: bool,
}

impl<'a, I, M> Tracker<'a, I, M>
where
    I: Clone + Send,
    M: Clone + Send + Debug + PartialEq + fmt::Display,
{
    pub fn new(
        sample_rate: i32,
        captured_output_dir: path::PathBuf,
        captured_date_format: String,
        command_receiver: mpsc::Receiver<Command<I, M>>,
        status_sender: mpsc::Sender<Status<I, M>>,
    ) -> Tracker<'a, I, M> {
        return Tracker {
            generator: generator::Generator::new(sample_rate),
            sample_rate,
            captured_output_dir,
            captured_date_format,
            command_receiver,
            status_sender,

            active_waveforms: Vec::new(),
            pending_waveforms: Vec::new(),

            send_current_buffer: false,
        };
    }

    fn process_captured<S>(
        &self,
        waveform: &waveform::Waveform<M, S>,
        out: &mut HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
    ) {
        use waveform::Waveform::*;
        match waveform {
            Const(_) | Time(_) | Noise | Fixed(_, _) => {
                return;
            }
            // TODO think harder about captures for the other sub-waveforms in these cases
            Fin { waveform, .. }
            | Reset {
                trigger: waveform, ..
            }
            | Alt {
                trigger: waveform, ..
            }
            | Marked { waveform, .. } => {
                self.process_captured(&*waveform, out);
            }
            Sine {
                frequency: a,
                phase: b,
                ..
            }
            | Append(a, b)
            | BinaryPointOp(_, a, b) => {
                self.process_captured(&*a, out);
                self.process_captured(&*b, out);
            }
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => {
                self.process_captured(waveform, out);
                for waveform in feed_forward.iter() {
                    self.process_captured(waveform, out);
                }
                for waveform in feedback.iter() {
                    self.process_captured(waveform, out);
                }
            }
            Captured {
                file_stem,
                waveform,
            } => {
                use hound::{SampleFormat, WavSpec, WavWriter};
                self.process_captured(&*waveform, out);
                if out.contains_key(file_stem) {
                    // TODO in theory we could check for this earlier
                    panic!("Captured waveform with duplicate file stem: {}", file_stem);
                }
                let datetime = chrono::Local::now().format(&self.captured_date_format);
                let file_name = format!("{}{}.wav", &file_stem, &datetime);
                let path = self.captured_output_dir.join(file_name);
                let file = std::fs::File::create(path).expect("Failed to create file");
                let spec = WavSpec {
                    channels: 1,
                    sample_rate: self.sample_rate as u32,
                    bits_per_sample: 32,
                    sample_format: SampleFormat::Float,
                };
                let writer = WavWriter::new(BufWriter::new(file), spec)
                    .expect("Failed to create WAV writer");
                out.insert(file_stem.clone(), writer);
            }
            Prior => (),
        }
    }
}

fn process_marked<I, M>(
    generator: &mut generator::Generator,
    sample_rate: f32,
    waveform_id: &I,
    start: Instant,
    waveform: &generator::Waveform<M>,
    out: &mut Vec<Mark<I, M>>,
) where
    M: Clone + Debug + fmt::Display,
    I: Clone,
{
    use waveform::Waveform::*;
    match waveform {
        Const(_) | Time(_) | Noise | Fixed(_, _) => {
            return;
        }
        // TODO Fin seems not quite right here, since its length might truncate any marks inside it
        Fin { waveform, .. }
        | Filter { waveform, .. }
        | Reset {
            trigger: waveform, ..
        }
        | Alt {
            trigger: waveform, ..
        }
        | Captured { waveform, .. } => {
            process_marked(
                generator,
                sample_rate,
                waveform_id,
                start,
                waveform.as_ref(),
                out,
            );
        }
        Sine {
            frequency, phase, ..
        } => {
            // TODO this is a little strange... but maybe correct?
            process_marked(
                generator,
                sample_rate,
                waveform_id,
                start,
                frequency.as_ref(),
                out,
            );
            process_marked(
                generator,
                sample_rate,
                waveform_id,
                start,
                phase.as_ref(),
                out,
            );
        }
        Append(a, b) => {
            process_marked(generator, sample_rate, waveform_id, start, &*a, out);
            let a_len = generator.length(
                &mut a.clone(),
                10 * sample_rate as usize, // XXX
            );
            let start = start + Duration::from_secs_f32(a_len as f32 / sample_rate as f32);
            process_marked(generator, sample_rate, waveform_id, start, b.as_ref(), out);
        }
        BinaryPointOp(_, a, b) => {
            process_marked(generator, sample_rate, waveform_id, start, &*a, out);
            process_marked(generator, sample_rate, waveform_id, start, &*b, out);
        }
        Marked { waveform, id } => {
            let len = generator.length(
                &mut waveform.clone(),
                10 * sample_rate as usize, // XXX
            );
            out.push(Mark {
                waveform_id: waveform_id.clone(),
                mark_id: id.clone(),
                start,
                duration: Duration::from_secs_f32(len as f32 / sample_rate as f32),
            });
            process_marked(generator, sample_rate, waveform_id, start, &*waveform, out);
        }
        Prior => (),
    }
}

impl<'a, I, M> audio::AudioCallback for Tracker<'a, I, M>
where
    I: Clone + Debug + Send + PartialEq,
    M: Clone + Debug + Send + PartialEq + fmt::Display,
{
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        // Assume that the callback is called far enough in advance of when the samples are
        // needed that we can use time equal to the length of the buffer. If that's true, then
        // the moment corresponding to the start of the buffer is the current time plus the length
        // of the buffer.
        let buffer_start =
            Instant::now() + Duration::from_secs_f32(out.len() as f32 / self.sample_rate as f32);
        // Check to see if we have any new commands
        self.empty_command_queue(buffer_start);

        let mut status_to_send = Status {
            buffer_start,
            marks: Vec::new(),
            tracker_load: None,
            allocations_per_sample: None,
            buffer: None,
        };

        // Now generate!
        let generate_start = Instant::now();
        let (finished, allocations) = self.generate(buffer_start, out);
        status_to_send.tracker_load = Some(
            self.sample_rate as f32 / (out.len() as f32 / generate_start.elapsed().as_secs_f32()),
        );
        status_to_send.allocations_per_sample = Some(allocations as f32 / self.sample_rate as f32);

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

impl<'a, I, M> Tracker<'a, I, M>
where
    I: Clone + Debug + Send + PartialEq,
    M: Clone + Debug + Send + PartialEq + fmt::Display,
{
    // buffer_start is the time corresponding to the beginning of the current buffer
    fn process_command(&mut self, command: Command<I, M>, buffer_start: Instant) {
        match command {
            Command::Play {
                id,
                waveform,
                start,
                repeat_every,
            } => {
                if let Some(duration) = repeat_every {
                    println!(
                        "Received command to play waveform {:?} at {:?} and every {:?}: {}",
                        id, start, duration, waveform
                    );
                } else {
                    println!(
                        "Received command to play waveform {:?} at {:?}: {}",
                        id, start, waveform
                    );
                }
                let start = start.unwrap_or(buffer_start);
                let mut marks = Vec::new();
                let waveform = generator::initialize_state(waveform);
                process_marked(
                    &mut self.generator,
                    self.sample_rate as f32,
                    &id,
                    start,
                    &waveform,
                    &mut marks,
                );
                self.pending_waveforms.push(PendingWaveform {
                    id,
                    waveform,
                    start,
                    repeat_every,
                    marks,
                });
                self.pending_waveforms.sort_by_key(|w| w.start);
            }
            Command::Modify {
                id,
                mark_id,
                waveform,
            } => {
                println!(
                    "Received command to replace mark {:?} in waveform {:?} with new waveform: {}",
                    mark_id, id, waveform
                );
                let waveform = generator::initialize_state(waveform);
                let mark_id = Some(mark_id);
                for active in &mut self.active_waveforms {
                    if active.id == id {
                        waveform::substitute(&mut active.waveform, &mark_id, &waveform);
                        println!("  new waveform is: {}", active.waveform);
                        // Recompute the marks
                        active.marks.clear();
                        process_marked(
                            &mut self.generator,
                            self.sample_rate as f32,
                            &active.id,
                            active.start,
                            // We need to re-initialize here because this waveform has moved forward
                            // from the beginning, so if we use `active.start` above, we need to actually
                            // restart here too.
                            &generator::initialize_state(active.waveform.clone()),
                            &mut active.marks,
                        );
                    }
                }
                for pending in &mut self.pending_waveforms {
                    if pending.id == id {
                        waveform::substitute(&mut pending.waveform, &mark_id, &waveform);
                        println!("  new waveform is: {}", pending.waveform);
                        // Recompute the marks
                        pending.marks.clear();
                        process_marked(
                            &mut self.generator,
                            self.sample_rate as f32,
                            &pending.id,
                            pending.start,
                            &pending.waveform,
                            &mut pending.marks,
                        );
                    }
                }
            }
            Command::RemovePending { id } => {
                println!("Received command to remove pending waveform {:?}", id);
                self.pending_waveforms.retain(|w| w.id != id);
            }
            Command::SendCurrentBuffer => {
                self.send_current_buffer = true;
            }
        }
    }

    fn empty_command_queue(&mut self, buffer_start: Instant) {
        loop {
            match self.command_receiver.try_recv() {
                Ok(command) => self.process_command(command, buffer_start),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(e) => println!("Error receiving command: {:?}", e),
            }
        }
    }

    /// Generate from pending waveforms and active waveforms, filling the out buffer.
    ///
    /// Returns the set of active waveforms that finished generating and the number of samples allocated.
    fn generate(
        &mut self,
        buffer_start: Instant,
        out: &mut [f32],
    ) -> (Vec<ActiveWaveform<I, M>>, usize) {
        let mut allocations: usize = 0;
        // We'll generate in segments based on the set of active waveforms at a given time
        let mut segment_start = buffer_start;
        let mut segment_length = out.len();

        // Keep track of any active waveforms that finish generating
        let mut finished = Vec::new();
        out.fill(0.0);

        let mut filled = 0; // How much of the out buffer we've filled so far
        while filled < out.len() {
            // Check to see if any pending waveform starts at or before segment_start. If so, promote
            // them active waveforms.
            while !self.pending_waveforms.is_empty() {
                if self.pending_waveforms[0].start <= segment_start {
                    let mut pending = self.pending_waveforms.remove(0);
                    /*
                    println!(
                        "Activating waveform {:?} with start {:?} at time {:?}",
                        pending.id, pending.start, segment_start
                    );
                    */
                    let mut waveform = generator::initialize_state(pending.waveform.clone());
                    let mut capture_state = HashMap::new();
                    self.process_captured(&waveform, &mut capture_state);
                    if pending.start < segment_start {
                        // If the pending waveform starts before the segment start, then we need to
                        // adjust the position.
                        let delta_samples = ((segment_start - pending.start).as_secs_f32()
                            * self.sample_rate as f32)
                            .round() as usize;
                        if delta_samples > 0 {
                            if delta_samples > 1 {
                                println!(
                                    "Adjusting waveform {:?} position by {} samples",
                                    pending.id, delta_samples
                                );
                            }
                            // We need to actually generate and discard these samples to make sure that any stateful
                            // waveforms are properly initialized.
                            let mut generator = generator::Generator::new(self.sample_rate);
                            let capture_state = RefCell::new(&mut capture_state);
                            generator.capture_state = Some(capture_state);
                            let mut tmp = vec![0.0; delta_samples];
                            allocations += delta_samples;
                            _ = generator.generate(&mut waveform, &mut tmp);
                            allocations += generator.allocations;
                        }
                    }
                    self.active_waveforms.push(ActiveWaveform {
                        id: pending.id.clone(),
                        start: pending.start,
                        waveform,
                        marks: pending.marks,
                        capture_state,
                    });

                    if let Some(repeat_every) = pending.repeat_every {
                        // If it's repeating, find the next start time that's after segment_start
                        pending.start = pending.start + repeat_every;
                        // Check to see if we've missed one or more repetitions.
                        while pending.start <= segment_start {
                            pending.start += repeat_every;
                            println!("Missed repetition of waveform {:?}...", pending.id);
                        }
                        /*
                        println!(
                            "Scheduling waveform {:?} to repeat at {:?}",
                            pending.id, pending.start,
                        );
                        */
                        pending.marks = Vec::new();
                        let waveform = generator::initialize_state(pending.waveform);
                        process_marked(
                            &mut self.generator,
                            self.sample_rate as f32,
                            &pending.id,
                            pending.start,
                            &waveform,
                            &mut pending.marks,
                        );
                        pending.waveform = waveform;
                        self.pending_waveforms.push(pending);
                        self.pending_waveforms.sort_by_key(|w| w.start);
                    }
                } else {
                    // Set the length of the current segment to the start of the next pending waveform
                    // We take the ceiling here to make sure that we don't create a segment that is shorter
                    // than the duration of a single sample.
                    segment_length = segment_length.min(
                        ((self.pending_waveforms[0].start - segment_start).as_secs_f32()
                            * self.sample_rate as f32)
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
                    Duration::from_secs_f32(segment_length as f32 / self.sample_rate as f32);
                segment_length = out.len() - filled;
                continue;
            }

            let mut i = 0;
            let mut tmp = vec![0.0; segment_length];
            allocations += segment_length;
            while i < self.active_waveforms.len() {
                let active = &mut self.active_waveforms[i];
                let len;
                {
                    let mut generator = generator::Generator::new(self.sample_rate);
                    let capture_state = RefCell::new(&mut active.capture_state);
                    generator.capture_state = Some(capture_state);
                    len = generator.generate(&mut active.waveform, &mut tmp);
                    allocations += generator.allocations;
                }
                if len > segment_length {
                    panic!(
                        "Generated more samples than desired: {} > {} for waveform id {:?}: {:?}",
                        len, segment_length, active.id, active.waveform
                    );
                }
                // Add the generated samples to the output buffer (which was pre-zeroed)
                for (j, &x) in tmp[..len].iter().enumerate() {
                    out[filled + j] += x;
                }
                if len < segment_length {
                    // If we didn't generate enough samples, then remove this waveform from the active list
                    /*
                    println!(
                        "Removing waveform {:?} at time {:?} (len = {}, segment_length = {})",
                        active.id,
                        segment_start
                            + Duration::from_secs_f32(tmp.len() as f32 / self.sample_rate as f32),
                        len,
                        segment_length,
                    );
                    */
                    let active = self.active_waveforms.remove(i);
                    finished.push(active);
                } else {
                    i += 1;
                }
            }
            filled += segment_length;
            segment_start +=
                Duration::from_secs_f32(segment_length as f32 / self.sample_rate as f32);
            segment_length = out.len() - filled;
        }
        return (finished, allocations);
    }
}
