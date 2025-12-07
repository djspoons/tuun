use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::BufWriter;
use std::path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

extern crate sdl2;
use sdl2::audio;

use crate::generator;
use crate::waveform;

pub enum Command<I> {
    Play {
        // A unique id for this waveform
        id: I,
        waveform: waveform::Waveform,
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
    MoveSlider {
        // The slider to set
        slider: waveform::Slider,
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
    // Marks for each active waveform as well as any pending waveforms; a mark may appear more
    // than once if a given waveform is both active and pending
    pub marks: Vec<Mark<I>>,
    // The current values of the sliders (as of buffer_start)
    pub slider_values: HashMap<waveform::Slider, f32>,
    // Some status updates will include the current buffer
    pub buffer: Option<Vec<f32>>,
    // The current tracker load, the ratio of sample frequency to samples generated per second
    pub tracker_load: Option<f32>,
}

struct ActiveWaveform<I>
where
    I: Clone,
{
    id: I,
    waveform: generator::Waveform,
    marks: Vec<Mark<I>>,
    position: usize,
    // Open files used by Captured waveforms
    capture_state: HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
}

#[derive(Debug, Clone)]
struct PendingWaveform<I> {
    id: I,
    waveform: waveform::Waveform,
    start: Instant,
    repeat_every: Option<Duration>,
    marks: Vec<Mark<I>>,
}

pub struct Tracker<I>
where
    I: Clone + Send,
{
    sample_frequency: i32,
    captured_output_dir: path::PathBuf,
    captured_date_format: String,
    command_receiver: mpsc::Receiver<Command<I>>,
    status_sender: mpsc::Sender<Status<I>>,

    // Persistent generation state
    active_waveforms: Vec<ActiveWaveform<I>>,
    pending_waveforms: Vec<PendingWaveform<I>>, // sorted by start time
    // Command state
    send_current_buffer: bool,
    slider_state: generator::SliderState,
}

impl<I> Tracker<I>
where
    I: Clone + Send,
{
    pub fn new(
        sample_frequency: i32,
        captured_output_dir: path::PathBuf,
        captured_date_format: String,
        command_receiver: mpsc::Receiver<Command<I>>,
        status_sender: mpsc::Sender<Status<I>>,
    ) -> Tracker<I> {
        return Tracker {
            sample_frequency,
            captured_output_dir,
            captured_date_format,
            command_receiver,
            status_sender,

            active_waveforms: Vec::new(),
            pending_waveforms: Vec::new(),

            send_current_buffer: false,
            slider_state: generator::SliderState {
                last_values: HashMap::new(),
                changes: HashMap::new(),
                buffer_length: 0,
                buffer_position: 0,
            },
        };
    }

    fn process_marked(
        &self,
        waveform_id: &I,
        start: Instant,
        waveform: &waveform::Waveform<
            generator::FilterState,
            generator::SinState,
            generator::ResState,
        >,
        out: &mut Vec<Mark<I>>,
    ) {
        use waveform::Waveform::*;
        match waveform {
            Const(_) | Time | Noise | Fixed(_) | Slider { .. } => {
                return;
            }
            // TODO Fin seems not quite right here, since its length might truncate any marks inside it
            Fin { waveform, .. }
            | Seq { waveform, .. }
            | Filter { waveform, .. }
            | Res {
                trigger: waveform, ..
            }
            | Alt {
                trigger: waveform, ..
            }
            | Captured { waveform, .. } => {
                self.process_marked(waveform_id, start, &*waveform, out);
            }
            Sin {
                frequency, phase, ..
            } => {
                // TODO this is a little strange... but maybe correct?
                self.process_marked(waveform_id, start, &*frequency, out);
                self.process_marked(waveform_id, start, &*phase, out);
            }
            Append(a, b) => {
                self.process_marked(waveform_id, start, &*a, out);
                let remaining = generator::Generator::new(self.sample_frequency).remaining(
                    &*a,
                    0,
                    10 * self.sample_frequency as usize, // XXX
                );
                let start = start
                    + Duration::from_secs_f32(remaining as f32 / self.sample_frequency as f32);
                self.process_marked(waveform_id, start, &*b, out);
            }
            BinaryPointOp(_, a, b) => {
                self.process_marked(waveform_id, start, &*a, out);
                // TODO do something about this constant for max
                let offset = generator::Generator::new(self.sample_frequency)
                    .offset(&*a, 10 * self.sample_frequency as usize);
                let start =
                    start + Duration::from_secs_f32(offset as f32 / self.sample_frequency as f32);
                self.process_marked(waveform_id, start, &*b, out);
            }
            Marked { waveform, id } => {
                let length = generator::Generator::new(self.sample_frequency).remaining(
                    &*waveform.clone(),
                    0,
                    10 * self.sample_frequency as usize, // XXX
                );
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

    fn process_captured<F, S, R>(
        &self,
        waveform: &waveform::Waveform<F, S, R>,
        out: &mut HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
    ) {
        use waveform::Waveform::*;
        match waveform {
            Const(_) | Time | Noise | Fixed(_) | Slider { .. } => {
                return;
            }
            Fin { waveform, .. }
            | Seq { waveform, .. }
            | Filter { waveform, .. }
            | Res {
                trigger: waveform, ..
            }
            | Alt {
                trigger: waveform, ..
            }
            | Marked { waveform, .. } => {
                self.process_captured(&*waveform, out);
            }
            Sin {
                frequency: a,
                phase: b,
                ..
            }
            | Append(a, b)
            | BinaryPointOp(_, a, b) => {
                self.process_captured(&*a, out);
                self.process_captured(&*b, out);
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
                    sample_rate: self.sample_frequency as u32,
                    bits_per_sample: 32,
                    sample_format: SampleFormat::Float,
                };
                let writer = WavWriter::new(BufWriter::new(file), spec)
                    .expect("Failed to create WAV writer");
                out.insert(file_stem.clone(), writer);
            }
        }
    }
}

impl<'a, I> audio::AudioCallback for Tracker<I>
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
            slider_values: self.slider_state.last_values.clone(),
            tracker_load: None,
            buffer: None,
        };

        // Now generate!
        let generate_start = Instant::now();
        let finished = self.generate(buffer_start, out);
        status_to_send.tracker_load = Some(
            self.sample_frequency as f32
                / (out.len() as f32 / generate_start.elapsed().as_secs_f32()),
        );

        // Update the slider values based on the changes
        for (slider, change) in self.slider_state.changes.iter() {
            let last_value = self.slider_state.last_values.remove(slider).unwrap_or(0.5);
            self.slider_state
                .last_values
                .insert(slider.clone(), (last_value + change).min(1.0).max(0.0));
        }
        self.slider_state.changes.clear();

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
                self.process_marked(
                    &id,
                    start,
                    &generator::initialize_state(waveform.clone()),
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
            Command::MoveSlider { slider, delta } => {
                self.slider_state
                    .changes
                    .entry(slider)
                    .and_modify(|v| *v += delta)
                    .or_insert(delta);
            }
        }
    }

    fn empty_command_queue(&mut self) {
        loop {
            match self.command_receiver.try_recv() {
                Ok(command) => self.process_command(command),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(e) => println!("Error receiving command: {:?}", e),
            }
        }
    }

    // Generate from pending waveforms and active waveforms, filling the out buffer.
    // Returns how many samples were generated, or None if the no samples were generated
    // along with the set of active waveforms that finished generating
    fn generate(&mut self, buffer_start: Instant, out: &mut [f32]) -> Vec<ActiveWaveform<I>> {
        // We'll generate in segments based on the set of active waveforms at a given time
        let mut segment_start = buffer_start;
        let mut segment_length = out.len();
        self.slider_state.buffer_length = out.len();

        // Keep track of any active waveforms that finish generating
        let mut finished = Vec::new();

        for x in out.iter_mut() {
            *x = 0.0;
        }
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
                    let waveform = generator::initialize_state(pending.waveform.clone());
                    let mut position = 0;
                    if pending.start < segment_start {
                        // If the pending waveform starts before the segment start, then we need to
                        // adjust the position.
                        let delta_samples = ((segment_start - pending.start).as_secs_f32()
                            * self.sample_frequency as f32)
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
                            let mut generator = generator::Generator::new(self.sample_frequency);
                            // TODO this is a little weird since we don't really care about sliders... but :shrug:
                            self.slider_state.buffer_position = 0;
                            generator.slider_state = Some(&self.slider_state);
                            // TODO do we want to capture here?
                            generator.capture_state = None;
                            let out = generator.generate(&waveform, position as i64, delta_samples);
                            position += out.len();
                        }
                    }
                    let mut capture_state = HashMap::new();
                    self.process_captured(&waveform, &mut capture_state);
                    self.active_waveforms.push(ActiveWaveform {
                        id: pending.id.clone(),
                        waveform,
                        marks: pending.marks,
                        position,
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
                        self.process_marked(
                            &pending.id,
                            pending.start,
                            &generator::initialize_state(pending.waveform.clone()),
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
                let tmp: Vec<f32>;
                {
                    let mut generator = generator::Generator::new(self.sample_frequency);
                    self.slider_state.buffer_position = filled;
                    generator.slider_state = Some(&self.slider_state);
                    let capture_state = RefCell::new(&mut active.capture_state);
                    generator.capture_state = Some(capture_state);
                    tmp = generator.generate(
                        &active.waveform,
                        active.position as i64,
                        segment_length,
                    );
                }
                if tmp.len() > segment_length {
                    panic!(
                        "Generated more samples than desired: {} > {} for waveform id {:?} at position {}: {:?}",
                        tmp.len(),
                        segment_length,
                        active.id,
                        active.position,
                        active.waveform
                    );
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
                    /*
                    println!(
                        "Removing waveform {:?} at position {} and time {:?}",
                        active.id,
                        active.position,
                        segment_start
                            + Duration::from_secs_f32(
                                tmp.len() as f32 / self.sample_frequency as f32
                            )
                    );
                    */
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
        }
        return finished;
    }
}
