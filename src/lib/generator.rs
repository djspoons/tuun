use std::cell::RefCell;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::io::BufWriter;

use fastrand;

use crate::waveform;

#[derive(Debug, Clone)]
pub struct FilterState {
    // The state of the filter, used to store previously generated samples and the position of the first of those
    // samples. Used as an input to feedback.
    previous_out: RefCell<(Vec<f32>, i64)>,
}

#[derive(Debug, Clone)]
pub struct SinState {
    // The phase of the sine wave at positions starting from the second element.
    previous_phases: RefCell<(Vec<f64>, i64)>,
}

#[derive(Debug, Clone)]
pub struct ResState {
    // The positions of previous resets, along with a range within which no reset occurs; both values are inclusive.
    previous_resets: RefCell<Vec<(i64, i64)>>,
}

pub struct SliderState {
    // The final values of sliders at the end of the last generate call
    pub last_values: HashMap<waveform::Slider, f32>,
    // Any changes to the sliders since the last generate call
    pub changes: HashMap<waveform::Slider, f32>,
    // The length of the current buffer being generated
    pub buffer_length: usize,
    // The position in the buffer where the next sample will be written
    pub buffer_position: usize,
}

/*
 * Generator converts waveforms into sequences of samples.
 */
pub struct Generator<'a> {
    sample_frequency: i32,
    pub slider_state: Option<&'a SliderState>,
    pub capture_state:
        Option<RefCell<&'a mut HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>>>,
}

pub type Waveform = waveform::Waveform<FilterState, SinState, ResState>;

// TODO add metrics for waveform expr depth and total ops

#[derive(Debug, Copy, Clone)]
enum MaybeOption<T> {
    Some(T),
    None,
    Maybe, // The value may or may not be present
}

const MAX_FILTER_LENGTH: usize = 1000; // Maximum length of the filter coefficients

impl<'a> Generator<'a> {
    // Create a new generator with the given sample frequency. Note that slider_state and capture_state must be set
    // before `generate` is called.
    pub fn new(sample_frequency: i32) -> Self {
        Generator {
            sample_frequency,
            slider_state: None,
            capture_state: None,
        }
    }

    // Generate a vector of samples up to `desired` length. `position` indicates where
    // the beginning of the result is relative to the start of the waveform. If fewer than
    // 'desired' samples are generated, that indicates that this waveform has finished (and
    // generate won't be called on it again). For waveforms other than Noise and Slider or
    // waveforms that contain them, the result is determined only by the arguments (this
    // function is pure). Note that position can be negative and that in that case, this
    // will always return at least desired.min(-position) samples (that is, waveforms are not
    // allowed to end before position 0).
    pub fn generate(&self, waveform: &Waveform, position: i64, desired: usize) -> Vec<f32> {
        use waveform::Waveform::*;
        if desired == 0 {
            return vec![];
        }
        match waveform {
            &Const(value) => {
                return vec![value; desired];
            }
            Time => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (i as i64 + position) as f32 / self.sample_frequency as f32;
                }
                return out;
            }
            Noise => {
                let mut out = vec![0.0; desired];
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                return out;
            }
            Fixed(samples) => {
                if position >= samples.len() as i64 {
                    return vec![];
                } else if position < 0 {
                    if desired <= -position as usize {
                        return vec![0.0; desired];
                    } else {
                        let mut out = vec![0.0; -position as usize];
                        out.extend_from_slice(
                            &samples[0..(desired - (-position as usize)).min(samples.len())],
                        );
                        return out;
                    }
                } else {
                    // position >= 0
                    let position = position as usize;
                    return samples.clone()[position..(position + desired).min(samples.len())]
                        .to_vec();
                }
            }
            Fin {
                waveform: inner_waveform,
                ..
            } => {
                let remaining = self.remaining(waveform, position, desired);
                return self.generate(inner_waveform, position, remaining);
            }
            Seq { waveform, .. } => {
                return self.generate(waveform, position, desired);
            }
            Append(a, b) => {
                // TODO can't we just generate and use the length of the result rather than calling remaining?
                let remaining = self.remaining(a, position, desired);
                if remaining >= desired {
                    return self.generate(a, position, desired);
                } else if remaining > 0 {
                    let mut out = self.generate(a, position, remaining);
                    out.extend(self.generate(b, 0, desired - out.len()));
                    return out;
                } else {
                    // We have remaining(a, position) == 0 so we need to figure out how far into b we are.
                    // Get the length of a, and use that to compute the position into b. If remaining is 0 then
                    // position can't be negative (since no waveform can end before position 0).
                    assert!(position >= 0);
                    let length = self.remaining(a, 0, desired + position as usize);
                    return self.generate(b, position - length as i64, desired);
                }
            }
            Sin {
                frequency,
                phase,
                state,
            } => {
                let mut previous_phases = state.previous_phases.borrow_mut();

                let mut position = position;
                let mut desired = desired;

                // The output samples
                let mut out: Vec<f32> = vec![];

                if position < 0 {
                    // Don't even try to use direct synthesis for negative positions.
                    // This is not as accurate, but it's simpler and... probably fine?
                    // Especially since we don't really hear sine waves for very negative positions.
                    // ... TODO unless it's not fine when Sin is used as a trigger for Res?

                    println!(
                        "Skipping direct synthesis for Sin at negative position {}, desired {}: Sin({}, {})",
                        position, desired, frequency, phase
                    );

                    let mut freqs =
                        self.generate(frequency, position, desired.min(-position as usize));
                    self.generate_sin_indirectly(position, &mut freqs, phase, &mut out);

                    if out.len() == desired {
                        return out;
                    }
                    position += out.len() as i64;
                    desired -= out.len();
                }

                let mut freqs = self.generate(frequency, position, desired);
                if position == 0 {
                    match *phase.as_ref() {
                        // TODO could also include Captured(Const) and Marked(Const)
                        waveform::Waveform::Const(initial_phase) => {
                            println!(
                                "Using direct synthesis for sine at position {}, desired {}: Sin({}, {})",
                                position, desired, frequency, phase
                            );

                            previous_phases.0.clear();
                            previous_phases.0.push(freqs[0] as f64);
                            previous_phases.1 = 0;

                            self.generate_sin_directly(
                                freqs,
                                initial_phase as f64,
                                position,
                                desired,
                                &mut previous_phases,
                                &mut out,
                            );
                            //println!("Generated Sin at position 0: {:?}", &out);
                            return out;
                        }
                        _ => {} // Fall through to the indirect case below
                    }
                } else if position >= previous_phases.1
                    && position < previous_phases.1 + previous_phases.0.len() as i64
                {
                    let initial_phase = previous_phases.0[(position - previous_phases.1) as usize];

                    println!(
                        "Using direct synthesis for Sin at position {}; found phase in previous_phases: {:?}: Sin({}, {})",
                        position, initial_phase, frequency, phase
                    );

                    self.generate_sin_directly(
                        freqs,
                        initial_phase as f64,
                        position,
                        desired,
                        &mut previous_phases,
                        &mut out,
                    );
                    //println!("Generated Sin at position {}: {:?}", position, &out);

                    return out;
                }
                // This case covers both non-constant phase and missing previous phase
                // TODO we could figure out what the phase is at an arbitrary position
                println!(
                    "Generating sine indirectly at position {}: Sin({}, {})",
                    position, frequency, phase
                );
                self.generate_sin_indirectly(position, &mut freqs, phase, &mut out);
                return out;
            }
            Filter {
                waveform,
                feed_forward,
                feedback,
                state,
            } => {
                /*
                println!(
                    "Generating Filter at position {}, desired {}, waveform: {:?}, feed_forward: {:?}, feedback: {:?}, state: {:?}",
                    position, desired, waveform, feed_forward, feedback, state
                );
                */
                // Generate the filter coefficients
                let feed_forward_out = self.generate(feed_forward, 0, MAX_FILTER_LENGTH);
                let feedback_out = self.generate(feedback, 0, MAX_FILTER_LENGTH);

                // The goal here is to make sure out.len() >= feedback_out.len() and to fill in the first
                // feedback_out.len() samples.
                let mut out = vec![];
                // previous_position is the position that we need to start reusing samples from
                let mut previous_position = position - feedback_out.len() as i64;
                let mut previous_out = state.previous_out.borrow_mut();
                if previous_position < 0 {
                    // We could generate from this waveform for negative positions... but we need to be a little
                    // careful and filling with zeros here has less impact than the equivalent case for Res.
                    out.resize(-previous_position as usize, 0.0);
                    previous_position = 0;
                }
                if previous_position >= previous_out.1
                    && previous_position < previous_out.1 + previous_out.0.len() as i64
                {
                    out.extend_from_slice(
                        &previous_out.0[(previous_position - previous_out.1) as usize
                            ..((previous_position - previous_out.1) as usize
                                + (feedback_out.len() - out.len()))],
                    );
                }
                if out.len() < feedback_out.len() {
                    panic!(
                        "Filter state has insufficient previous output at position {}",
                        position
                    );
                }

                // The goal here is to get length of inner_out to be = desired + feed_forward_out.len() - 1
                // if possible; if it's shorter then the output will be shorter too.
                let inner_out = self.generate(
                    waveform,
                    position - (feed_forward_out.len() as i64 - 1),
                    desired + (feed_forward_out.len() - 1),
                );
                if inner_out.len() < feed_forward_out.len() {
                    return vec![]; // Not enough input to generate any output
                }
                // Set the output length based on the size of the inner waveform
                out.resize(
                    inner_out.len() - (feed_forward_out.len() - 1) + feedback_out.len(),
                    0.0,
                );

                // Run the filter!!
                for i in feedback_out.len()..out.len() {
                    for (j, &ff) in feed_forward_out.iter().enumerate() {
                        out[i] += ff
                            * inner_out[i - feedback_out.len() + (feed_forward_out.len() - 1) - j];
                    }
                    for (j, &fb) in feedback_out.iter().enumerate() {
                        out[i] -= fb * out[i - j - 1];
                    }
                }

                // Save up to MAX_FILTER_LENGTH samples for the next call to generate. Even though we will only need
                // feedback_out.len() samples, other filters that this one is nested within may need more.
                // First remove any samples before position (padding or previously generated)
                out.drain(0..feedback_out.len());
                if out.len() > MAX_FILTER_LENGTH {
                    previous_out.0 = out[out.len() - MAX_FILTER_LENGTH..].to_vec();
                } else {
                    previous_out.0.extend(out.iter());
                    let len = previous_out.0.len().saturating_sub(MAX_FILTER_LENGTH);
                    previous_out.0.drain(0..len);
                }
                previous_out.1 = position + out.len() as i64 - previous_out.0.len() as i64;
                return out;
            }
            BinaryPointOp(op, a, b) => {
                use waveform::Operator;
                let desired = match op {
                    Operator::Multiply | Operator::Divide => {
                        // We need to make sure we generate a length based on the shorter waveform.
                        self.remaining(waveform, position, desired)
                    }
                    _ => desired,
                };
                let op = match op {
                    Operator::Add => std::ops::Add::add,
                    Operator::Subtract => std::ops::Sub::sub,
                    Operator::Multiply => std::ops::Mul::mul,
                    Operator::Divide => |a: f32, b: f32| {
                        if b == 0.0 { 0.0 } else { a / b }
                    },
                };
                return self.generate_binary_op(op, a, b, position, desired);
            }
            Res {
                trigger,
                waveform,
                state,
            } => {
                // XXX Ugh... this constant...
                let max_reset_lookback: i64 = self.sample_frequency as i64 * 10000; // Maximum number of samples to look back for a reset trigger
                // TODO think about all of these unwrap_ors
                // TODO generate the trigger in blocks?
                // TODO think about the interaction of this non-monotone position with Filter

                let mut previous_resets = state.previous_resets.borrow_mut();
                /*
                println!(
                    "previous_resets when position = {}: {:?}",
                    position, previous_resets
                );
                */

                // First go back and find the most recent trigger before position.
                let mut last_trigger_position = position;
                let trigger_out = self.generate(trigger, position, 1);
                if trigger_out.is_empty() {
                    return vec![];
                }
                let mut last_signum = trigger_out[0].signum();
                'find_last_trigger_position: loop {
                    for (start, end) in previous_resets.iter_mut() {
                        if last_trigger_position >= *start && last_trigger_position <= *end {
                            /*
                            println!(
                                "Found previous reset during init for position {} at {} (ended with {}, now ending with {})",
                                last_trigger_position, start, end, position
                            );
                            */
                            last_trigger_position = *start;
                            last_signum = 1.0;
                            *end = position;
                            break 'find_last_trigger_position;
                        }
                    }

                    if position - last_trigger_position > max_reset_lookback {
                        panic!(
                            "No reset trigger found within {} samples before position {} in waveform {:?}",
                            max_reset_lookback, position, trigger
                        );
                    }
                    let trigger_out = self.generate(trigger, last_trigger_position - 1, 1);
                    let new_signum = trigger_out.get(0).unwrap_or(&0.0).signum();
                    if last_signum >= 0.0 && new_signum < 0.0 {
                        /*
                        println!(
                            "Adding new reset during init as ({}, {})",
                            last_trigger_position, position
                        );
                        */
                        previous_resets.push((last_trigger_position, position));
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
                    // Set to true if a restart will be triggered before desired is reached
                    let mut reset_inner_position = false;
                    let mut inner_desired = trigger_out.len() - generated;

                    for (i, &x) in trigger_out[generated..].iter().enumerate() {
                        if last_signum < 0.0 && x >= 0.0 {
                            inner_desired = i;
                            reset_inner_position = true;
                            last_signum = x.signum();
                            // Update previous_resets
                            if i > 0 {
                                let mut found = false;
                                for (start, end) in previous_resets.iter_mut() {
                                    if position + generated as i64 >= *start
                                        && position + generated as i64 <= *end
                                    {
                                        /*
                                        println!(
                                            "Updating previous reset during generate for position {} at {} (ended with {}, now ending at {})",
                                            position + generated as i64,
                                            start,
                                            end,
                                            position + (generated + i) as i64 - 1
                                        );
                                        */
                                        *end = position + (generated + i) as i64 - 1;
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    /*
                                    println!(
                                        "Adding new reset during generate as ({}, {})",
                                        position + generated as i64,
                                        position + (generated + i) as i64
                                    );
                                    */
                                    previous_resets.push((
                                        position + generated as i64,
                                        position + (generated + i) as i64 - 1,
                                    ));
                                }
                            }
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
                    if reset_inner_position {
                        inner_position = 0;
                    } else {
                        inner_position += inner_desired as i64;
                        // Update the previous_resets
                        let mut found = false;
                        for (start, end) in previous_resets.iter_mut() {
                            if position + generated as i64 >= *start
                                && position + generated as i64 <= *end
                            {
                                /*
                                println!(
                                    "Updating previous reset after generate for position {} at {} (ended with {}, now ending at {})",
                                    position + (generated + inner_desired) as i64,
                                    start,
                                    end,
                                    position + (generated + inner_desired) as i64 - 1
                                );
                                */
                                *end = position + (generated + inner_desired) as i64 - 1;
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            /*
                            println!(
                                "Adding new reset after generate as ({}, {})",
                                position + generated as i64,
                                position + (generated + inner_desired) as i64 - 1
                            );
                            */
                            previous_resets.push((
                                position + generated as i64,
                                position + (generated + inner_desired) as i64 - 1,
                            ));
                        }
                    }
                    generated += inner_desired;
                }

                // TODO think harder about what a reasonable number is here
                if previous_resets.len() > 10 {
                    let length = previous_resets.len();
                    previous_resets.drain(..length - 10);
                }
                return out;
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let mut out = self.generate(trigger, position, desired);
                // TODO better to generate just one of the two alternates based on runs of signum?
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
            Slider(slider) => {
                if self.slider_state.is_none() {
                    println!("Warning: Slider waveform used, but no slider state set");
                    return vec![0.0; desired];
                }
                let slider_state = self.slider_state.unwrap();
                let last_value = slider_state
                    .last_values
                    .get(&slider)
                    .cloned()
                    .unwrap_or(0.5);
                let change = slider_state.changes.get(&slider).cloned().unwrap_or(0.0);
                // Use a linear interpolation between the last value and the change. This is almost right, but if
                // a slider waveform is used in a binary op, then buffer_position + position might not be large
                // enough.
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = last_value
                        + change
                            * ((slider_state.buffer_position as f32
                                / slider_state.buffer_length as f32)
                                + i as f32 / desired as f32)
                                .min(1.0)
                                .max(0.0);
                }
                /*
                println!(
                    "Slider {:?}, last_value: {}, change: {}, average value: {}",
                    slider,
                    last_value,
                    change,
                    (out[0] + out[out.len() - 1]) / 2.0
                );
                println!("Slider {:?} values: {:?}", slider, out);
                */
                return out;
            }
            Marked { waveform, .. } => {
                return self.generate(waveform, position, desired);
            }
            Captured {
                file_stem,
                waveform,
            } => {
                // TODO think through this again
                //  - capture_state was set incorrectly when advancing position (i.e., when a waveform missed its start time)
                //  - we used to not generate the inner waveform when that was set... or was it unset?
                //  - that means Sin wouldn't set previous_phases at position 0
                let out = self.generate(waveform, position, desired);
                if self.capture_state.is_none() {
                    // This occurs, for example, in cases where we need to advance the position of a waveform
                    return out;
                }
                match self
                    .capture_state
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .get_mut(file_stem)
                {
                    Some(writer) => {
                        for x in out.iter() {
                            if let Err(e) = writer.write_sample(*x) {
                                eprintln!("Error writing sample for {}: {}", file_stem, e);
                            }
                        }
                    }
                    None => {
                        panic!("No open file for captured waveform {}", file_stem);
                    }
                }
                return out;
            }
        }
    }

    fn generate_sin_indirectly(
        &self,
        position: i64,
        freqs: &mut Vec<f32>,
        phase: &Waveform,
        out: &mut Vec<f32>,
    ) {
        let phases = self.generate(phase, position, freqs.len());
        if freqs.len() < phases.len() {
            freqs.truncate(phases.len());
        }
        for (i, &x) in freqs.iter().enumerate() {
            let phase = (x as f64 * ((i as i64 + position) as f64 / self.sample_frequency as f64)
                + phases[i] as f64)
                .rem_euclid(f64::consts::TAU);
            out.push(phase.sin() as f32);
        }
    }

    fn generate_sin_directly(
        &self,
        freqs: Vec<f32>,
        initial_phase: f64,
        position: i64,
        desired: usize,
        previous_phases: &mut (Vec<f64>, i64),
        out: &mut Vec<f32>,
    ) {
        let previous_phase_start_position;
        // Figure out which phases we need to save for future calls to generate
        if desired >= MAX_FILTER_LENGTH + 1 {
            previous_phase_start_position =
                position + desired as i64 - (MAX_FILTER_LENGTH as i64 + 1);
            previous_phases.0.clear();
            // We'll starting computing phases at previous_phase_start_position, when we'll add the
            // phase at the end of the loop to the set; this corresponds to the phase used at the
            // beginning of computing `previous_phase_start_position + 1`.
            previous_phases.1 = previous_phase_start_position + 1;
        } else {
            // Note that this means we will add a phase to the set starting at the end of computing
            // `previous_phases.1 + previous_phases.0.len() - 1`, which is the phase used at the
            // beginning of `previous_phases.1 + previous_phases.0.len()`.
            previous_phase_start_position = previous_phases.1 + previous_phases.0.len() as i64 - 1;
            // Trim any old phases
            if position + desired as i64 - previous_phases.1 > MAX_FILTER_LENGTH as i64 {
                let len = (position + desired as i64 - previous_phases.1 - MAX_FILTER_LENGTH as i64)
                    as usize;
                previous_phases.0.drain(0..len);
                previous_phases.1 += len as i64;
            }
        }

        let mut phase = initial_phase;
        for (i, x) in freqs.iter().enumerate() {
            out.push(phase.sin() as f32);
            let phase_inc = *x as f64 / self.sample_frequency as f64;
            phase = (phase + phase_inc).rem_euclid(f64::consts::TAU);

            if position + i as i64 >= previous_phase_start_position {
                previous_phases.0.push(phase);
            }
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
        position: i64,
        desired: usize,
    ) -> Vec<f32> {
        let mut left = self.generate(a, position, desired);
        // Note on negative positions and offsets: we assume that if a's offset is non-zero that the
        // intention is that no negative positions should be generated for b. (This is consistent with
        // how Seq is replaced by appending b to a finite Const waveform whose value is the identity for
        // the op.). If a's offset is zero, then we do generate negative positions for b.

        // Also note that if we assume that offset(a) == 0 this is all much simpler, but I'm keeping this
        // code around as a test of waveform optimizations (including removing Seq's).

        // Calculate the maximum offset of `a` that is relevant for position and desired. This is the sum
        // of positive portion of position (if any) and the positive portion of desired (again, if any).
        let positive_position = if position < 0 { 0 } else { position as usize };
        let positive_desired = if position < 0 {
            if desired <= (-position) as usize {
                0
            } else {
                desired - (-position) as usize
            }
        } else {
            desired
        };
        let offset = self.offset(&a, positive_position + positive_desired);

        if offset > 0 {
            if offset >= positive_position + positive_desired {
                // offset should never be > max_offset so offset == max_offset, which means that b shouldn't
                // generate here. Make sure the left side is long enough so that we get another chance to generate
                // the right-hand side.
                left.resize(desired, 0.0);
                return left;
            }
            // offset < positive_position + positive_desired
            // There is an overlap between the desired portion and the right waveform, we have two cases (below)
            // to handle. But first, if the left side is shorter than the offset then extend it.
            // TODO I think this might be simplified by combining it with the two cases below.
            if position < 0 {
                left.resize((offset + (-position) as usize).max(desired), 0.0);
            } else if position + left.len() as i64 <= offset as i64 {
                left.resize(offset - position as usize, 0.0);
            }

            if positive_position <= offset {
                // ... 1) the right waveform starts at or after positive_position
                let right = self.generate(b, 0, positive_desired - (offset - positive_position));
                // Merge the overlapping portion
                // The index into left here could also be offset - positive_position - negative_position
                for (i, x) in left[(offset as i64 - position) as usize..]
                    .iter_mut()
                    .enumerate()
                {
                    if i >= right.len() {
                        break;
                    }
                    *x = op(*x, right[i]);
                }
                // If the left side is shorter than the right, than append.
                // Another way of writing this would be to ask if:
                //          right.len() + offset > left.len() + positive_position - negative_position
                if (right.len() + offset) as i64 > left.len() as i64 + position {
                    left.extend_from_slice(&right[(left.len() + positive_position - offset)..]);
                }
            } else {
                // ... 2) the right waveform starts before positive_position
                // Since offset >= 0 and offset < positive_position, we know that positive_position == position
                // and that left has no samples corresponding to negative positions.
                let right = self.generate(b, (positive_position - offset) as i64, desired);
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
        } else {
            // offset == 0, so we use the same position for both a and b
            match b {
                // In this branch (which is always taken if we've removed Seq's), check to see if we can
                // avoid generating the right-hand side and instead just apply the op directly.
                // TODO could also check f against the identity of op and skip the loop here
                Waveform::Const(f) => {
                    for x in left.iter_mut() {
                        *x = op(*x, *f);
                    }
                    if desired > left.len() {
                        left.resize(desired, op(0.0, *f));
                    }
                }
                _ => {
                    let right = self.generate(b, position - offset as i64, desired);
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
        }

        return left;
    }

    // Returns the number of samples that `waveform` will generate starting from `position` or `max`, whichever is
    // smaller. When `position` is zero, this is the length of the waveform (or, again, `max`).
    pub fn remaining(&self, waveform: &Waveform, position: i64, max: usize) -> usize {
        use waveform::Operator;
        use waveform::Waveform::*;
        match waveform {
            Const { .. } => max,
            Time => max,
            Noise => max,
            Fixed(samples) => {
                if position >= samples.len() as i64 {
                    0
                } else if position < 0 {
                    (samples.len() + (-position) as usize).min(max)
                } else {
                    (samples.len() - position as usize).min(max)
                }
            }
            Fin { length, waveform } => {
                // This is a little subtle, since Fin is not supposed to make waveforms longer. In particular, for
                // optimizations that move Fin outside of a DotProduct, we need to check the length of the inner
                // waveform.
                // TODO figure out which of `length` or `length(waveform)` is cheaper and do that first.
                let inner = self.remaining(waveform, position, max);
                return match self.greater_or_equals_at(&length, 0.0, position, max) {
                    MaybeOption::Some(new_position) => new_position.min(inner),
                    MaybeOption::None => inner,
                    MaybeOption::Maybe => {
                        println!(
                            "Warning: unable to determine length of Fin length cheaply, generating samples for: {:?}",
                            length
                        );
                        let out = self.generate(length, position, inner);
                        for (i, &x) in out.iter().enumerate() {
                            if x >= 0.0 {
                                return i;
                            }
                        }
                        return inner;
                    }
                };
            }
            Seq { waveform, .. } | Filter { waveform, .. } => {
                self.remaining(waveform, position, max)
            }
            Append(a, b) => {
                let a_remaining = self.remaining(a, position, max);
                if a_remaining == max {
                    a_remaining
                } else if a_remaining > 0 {
                    a_remaining + self.remaining(b, 0, max - a_remaining)
                } else {
                    // a_remaining == 0 and max > 0, so position can't be negative (since no waveform can
                    // end before position 0).
                    assert!(position >= 0);
                    let a_length = self.remaining(a, 0, position as usize + max);
                    self.remaining(b, position - a_length as i64, max)
                }
            }
            Sin {
                frequency, phase, ..
            } => {
                let freq_remaining = self.remaining(frequency, position, max);
                let phase_remaining = self.remaining(phase, position, max);
                freq_remaining.min(phase_remaining)
            }
            BinaryPointOp(op, a, b) => {
                let from_a = self.remaining(a, position, max);
                let positive_position = if position < 0 { 0 } else { position as usize };
                let positive_max = if position < 0 {
                    if max <= (-position) as usize {
                        0
                    } else {
                        max - (-position) as usize
                    }
                } else {
                    max
                };
                let offset = self.offset(a, positive_position + positive_max);
                let from_b = if offset > 0 {
                    if offset >= positive_position + positive_max {
                        positive_max
                    } else if positive_position < offset {
                        // positive_position + positive_max > offset
                        // positive_max > offset - positive_position
                        self.remaining(b, 0, positive_max - (offset - positive_position))
                            + (offset - positive_position)
                    } else {
                        // positive_position >= offset
                        self.remaining(b, (positive_position - offset) as i64, max)
                    }
                } else {
                    // offset == 0
                    // In this case, position is the same for both a and b
                    self.remaining(b, position, max)
                };
                match op {
                    Operator::Add | Operator::Subtract => from_a.max(from_b),
                    Operator::Multiply | Operator::Divide => from_a.min(from_b),
                }
            }
            Res { trigger, .. } | Alt { trigger, .. } => self.remaining(trigger, position, max),
            Slider { .. } => max,
            Marked { waveform, .. } | Captured { waveform, .. } => {
                self.remaining(waveform, position, max)
            }
        }
    }

    // If `waveform` will be greater than or equal to `value` at some point between `position` and `position + max`,
    // return Some of the difference between that point and position, None if `waveform` will not be greater
    // than or equal in that range, or Maybe if that can't be determined cheaply.
    fn greater_or_equals_at(
        &self,
        waveform: &Waveform,
        value: f32,
        position: i64,
        max: usize,
    ) -> MaybeOption<usize> {
        use waveform::Operator;
        use waveform::Waveform::{Append, BinaryPointOp, Const, Time};
        match waveform {
            Const(v) if *v >= value => MaybeOption::Some(0),
            Const(_) => MaybeOption::None,
            Time => {
                let current_value = position as f32 / self.sample_frequency as f32;
                if current_value >= value {
                    MaybeOption::Some(0)
                } else {
                    let target_position = (value * self.sample_frequency as f32).ceil() as i64;
                    MaybeOption::Some(max.min((target_position - position) as usize))
                }
            }
            Append(a, b) => {
                match self.greater_or_equals_at(a, value, position, max) {
                    MaybeOption::Some(size) => MaybeOption::Some(size),
                    MaybeOption::None => {
                        let a_remaining = self.remaining(a, position, max);
                        if a_remaining == max {
                            // We didn't reach the end of a, so b isn't relevant yet
                            MaybeOption::None
                        } else if a_remaining == 0 {
                            // If a_remaining == 0 but max > 0, then position can't be negative (since no waveform can
                            // end before position 0).
                            assert!(position >= 0);
                            // Find how far we are into b
                            let a_length = self.remaining(a, 0, position as usize + max);
                            self.greater_or_equals_at(
                                b,
                                value,
                                position - a_length as i64,
                                max - a_length,
                            )
                        } else {
                            // 0 < a_remaining < max, so b starts between position and position + max
                            self.greater_or_equals_at(b, value, 0, max - a_remaining)
                        }
                    }
                    MaybeOption::Maybe => MaybeOption::Maybe,
                }
            }
            BinaryPointOp(op @ (Operator::Add | Operator::Subtract), a, b) => {
                use waveform::Operator::{Add, Subtract};
                // If a has an offset, we'd need to handle the overlapping and non-overlapping parts
                // separately, and since we don't expect this to be called on non-optimized waveforms (except
                // in tests), just give up.
                if self.offset(a, max) != 0 {
                    return MaybeOption::Maybe;
                }
                match (op, a.as_ref(), b.as_ref()) {
                    // TODO need to consider Sliders and constant functions of Sliders as const
                    (Add, Const(va), Const(vb)) if va + vb >= value => MaybeOption::Some(0),
                    (Add, Const(_), Const(_)) => MaybeOption::None,
                    (Add, Const(va), _) => self.greater_or_equals_at(b, value - va, position, max),
                    (Add, _, Const(vb)) => self.greater_or_equals_at(a, value - vb, position, max),

                    (Subtract, Const(va), Const(vb)) if va - vb >= value => MaybeOption::Some(0),
                    (Subtract, Const(_), Const(_)) => MaybeOption::None,
                    (Subtract, _, Const(vb)) => {
                        self.greater_or_equals_at(a, value + vb, position, max)
                    }

                    _ => MaybeOption::Maybe,
                }
            }
            _ => MaybeOption::Maybe,
        }
    }

    // Returns the offset at which the next waveform should start (in samples). This is determined entirely by
    // `waveform` (this function is pure).
    pub fn offset(&self, waveform: &Waveform, max: usize) -> usize {
        use waveform::Waveform::*;
        match waveform {
            Const { .. } | Time | Noise | Fixed(_) => 0,
            Fin { waveform, .. } => self.offset(waveform, max),
            Seq { offset, .. } => match self.greater_or_equals_at(&offset, 0.0, 0, max) {
                MaybeOption::Some(new_position) => new_position,
                MaybeOption::None => max,
                MaybeOption::Maybe => {
                    panic!(
                        "Unable to determine offset of Seq offset cheaply: {:?}",
                        offset
                    );
                }
            },
            Append(a, b) => (self.offset(a, max) + self.offset(b, max)).min(max),
            Sin {
                frequency, phase, ..
            } => self.offset(frequency, max) + self.offset(phase, max),
            Filter { waveform, .. } => self.offset(waveform, max),
            BinaryPointOp(_, a, b) => {
                let a_offset = self.offset(a, max);
                if a_offset == max {
                    return max;
                }
                let b_offset = self.offset(b, max - a_offset);
                return a_offset + b_offset;
            }
            Res { trigger, .. } | Alt { trigger, .. } => self.offset(trigger, max),
            Slider { .. } => 0,
            Marked { waveform, .. } | Captured { waveform, .. } => self.offset(waveform, max),
        }
    }
}

pub fn initialize_state(waveform: waveform::Waveform) -> Waveform {
    use waveform::Waveform::*;
    match waveform {
        Const(value) => Const(value),
        Time => Time,
        Noise => Noise,
        Fixed(samples) => Fixed(samples),
        Fin { length, waveform } => Fin {
            length: Box::new(initialize_state(*length)),
            waveform: Box::new(initialize_state(*waveform)),
        },
        Seq { offset, waveform } => Seq {
            offset: Box::new(initialize_state(*offset)),
            waveform: Box::new(initialize_state(*waveform)),
        },
        Append(a, b) => Append(
            Box::new(initialize_state(*a)),
            Box::new(initialize_state(*b)),
        ),
        Sin {
            frequency, phase, ..
        } => Sin {
            frequency: Box::new(initialize_state(*frequency)),
            phase: Box::new(initialize_state(*phase)),
            state: SinState {
                previous_phases: RefCell::new((vec![], 0)),
            },
        },
        // For Filter, we need to set the state to an empty FilterState
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => Filter {
            waveform: Box::new(initialize_state(*waveform)),
            feed_forward: Box::new(initialize_state(*feed_forward)),
            feedback: Box::new(initialize_state(*feedback)),
            state: FilterState {
                previous_out: RefCell::new((vec![], 0)),
            },
        },
        BinaryPointOp(op, a, b) => BinaryPointOp(
            op,
            Box::new(initialize_state(*a)),
            Box::new(initialize_state(*b)),
        ),
        Res {
            trigger, waveform, ..
        } => Res {
            trigger: Box::new(initialize_state(*trigger)),
            waveform: Box::new(initialize_state(*waveform)),
            state: ResState {
                previous_resets: RefCell::new(vec![]),
            },
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(initialize_state(*trigger)),
            positive_waveform: Box::new(initialize_state(*positive_waveform)),
            negative_waveform: Box::new(initialize_state(*negative_waveform)),
        },
        Slider(slider) => Slider(slider),
        Marked { id, waveform } => Marked {
            id,
            waveform: Box::new(initialize_state(*waveform)),
        },
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(initialize_state(*waveform)),
        },
    }
}

pub fn remove_state<F, S, R>(w: waveform::Waveform<F, S, R>) -> waveform::Waveform
where
    F: Clone + Debug,
    R: Clone + Debug,
    S: Clone + Debug,
{
    use waveform::Waveform::*;
    match w {
        Const(value) => Const(value),
        Time => Time,
        Noise => Noise,
        Fixed(samples) => Fixed(samples),
        Fin { length, waveform } => Fin {
            length: Box::new(remove_state(*length)),
            waveform: Box::new(remove_state(*waveform)),
        },
        Seq { offset, waveform } => Seq {
            offset: Box::new(remove_state(*offset)),
            waveform: Box::new(remove_state(*waveform)),
        },
        Append(a, b) => Append(Box::new(remove_state(*a)), Box::new(remove_state(*b))),
        Sin {
            frequency, phase, ..
        } => Sin {
            frequency: Box::new(remove_state(*frequency)),
            phase: Box::new(remove_state(*phase)),
            state: (),
        },
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => Filter {
            waveform: Box::new(remove_state(*waveform)),
            feed_forward: Box::new(remove_state(*feed_forward)),
            feedback: Box::new(remove_state(*feedback)),
            state: (),
        },
        BinaryPointOp(op, a, b) => {
            BinaryPointOp(op, Box::new(remove_state(*a)), Box::new(remove_state(*b)))
        }
        Res {
            trigger, waveform, ..
        } => Res {
            trigger: Box::new(remove_state(*trigger)),
            waveform: Box::new(remove_state(*waveform)),
            state: (),
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(remove_state(*trigger)),
            positive_waveform: Box::new(remove_state(*positive_waveform)),
            negative_waveform: Box::new(remove_state(*negative_waveform)),
        },
        Slider(slider) => Slider(slider),
        Marked { id, waveform } => Marked {
            id,
            waveform: Box::new(remove_state(*waveform)),
        },
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(remove_state(*waveform)),
        },
    }
}

// Replaces parts of `waveform` that can be precomputed with their precomputed Fixed versions. Notably,
// infinite waveforms and waveforms that depend on or have dynamic behavior (Slider, Marked, Captured)
// cannot be precomputed. This should be called after remove_seq.
//
// N.B. This isn't currently safe as values at negative positions aren't considered; for example, time is
// negative before zero, but a fixed waveform is always zero before its start.
pub fn precompute(generator: &Generator, waveform: Waveform) -> Waveform {
    enum Result {
        Precomputed(Vec<f32>),
        Infinite(Waveform),
        Dynamic(Waveform),
    }

    impl Into<Waveform> for Result {
        fn into(self) -> Waveform {
            match self {
                Result::Precomputed(v) => Waveform::Fixed(v),
                Result::Infinite(w) => w,
                Result::Dynamic(w) => w,
            }
        }
    }

    fn precompute_internal(generator: &Generator, waveform: Waveform) -> Result {
        use Result::*;
        use waveform::Operator;
        use waveform::Waveform::*;

        // TODO how do we feel about all of the usize::MAX here?
        // TODO there's slightly more repetition than I'd like... for example, lots of "if the child is Dynamic,
        // then parent is Dynamic"
        // Maybe Precomputed should also be a waveform?

        match waveform {
            Const(_) | Time | Noise => Infinite(waveform),
            Fixed(v) => Precomputed(v),
            Fin { length, waveform } => match precompute_internal(generator, *waveform) {
                Precomputed(v) => Precomputed(generator.generate(
                    &Fin {
                        length,
                        waveform: Box::new(Fixed(v)),
                    },
                    0,
                    usize::MAX,
                )),
                Infinite(waveform) => Precomputed(generator.generate(
                    &Fin {
                        length,
                        waveform: Box::new(waveform),
                    },
                    0,
                    usize::MAX,
                )),
                Dynamic(waveform) => {
                    println!(
                        "Cannot precompute Fin because inner waveform is dynamic: {:?}",
                        &waveform
                    );
                    Dynamic(Fin {
                        length,
                        waveform: Box::new(waveform),
                    })
                }
            },
            Seq { .. } => {
                panic!("Seq should have been replaced by replace_seq before precompute is called");
            }
            Append(a, b) => match (
                precompute_internal(generator, *a),
                precompute_internal(generator, *b),
            ) {
                (Precomputed(a), Precomputed(b)) => {
                    let v = [a, b].concat();
                    Precomputed(v)
                }
                (Infinite(a), Infinite(b)) => Infinite(Append(Box::new(a), Box::new(b))),
                (Infinite(a), Precomputed(v)) => Infinite(Append(Box::new(a), Box::new(Fixed(v)))),
                (Precomputed(v), Infinite(b)) => Infinite(Append(Box::new(Fixed(v)), Box::new(b))),
                // At least one is Dynamic
                (a, b) => Dynamic(Append(Box::new(a.into()), Box::new(b.into()))),
            },
            Sin {
                frequency,
                phase,
                state,
            } => match (
                precompute_internal(generator, *frequency),
                precompute_internal(generator, *phase),
            ) {
                (Precomputed(f), Precomputed(p)) => Precomputed(generator.generate(
                    &Sin {
                        frequency: Box::new(Fixed(f)),
                        phase: Box::new(Fixed(p)),
                        state,
                    },
                    0,
                    usize::MAX,
                )),
                (Infinite(f), Infinite(p)) => Infinite(Sin {
                    frequency: Box::new(f),
                    phase: Box::new(p),
                    state,
                }),
                (Infinite(f), Precomputed(p)) => Infinite(Sin {
                    frequency: Box::new(f),
                    phase: Box::new(Fixed(p)),
                    state,
                }),
                (Precomputed(f), Infinite(p)) => Infinite(Sin {
                    frequency: Box::new(Fixed(f)),
                    phase: Box::new(p),
                    state,
                }),
                // At least one is Dynamic
                (f, p) => Dynamic(Sin {
                    frequency: Box::new(f.into()),
                    phase: Box::new(p.into()),
                    state,
                }),
            },
            BinaryPointOp(op, a, b) => match (
                precompute_internal(generator, *a),
                precompute_internal(generator, *b),
            ) {
                (Precomputed(a), Precomputed(b)) => Precomputed(generator.generate(
                    &BinaryPointOp(op, Box::new(Fixed(a)), Box::new(Fixed(b))),
                    0,
                    usize::MAX,
                )),
                (Infinite(a), Infinite(b)) => Infinite(BinaryPointOp(op, Box::new(a), Box::new(b))),
                (Infinite(a), Precomputed(v)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(a), Box::new(Fixed(v))))
                    }
                    Operator::Multiply | Operator::Divide => Precomputed(generator.generate(
                        &BinaryPointOp(op, Box::new(a), Box::new(Fixed(v))),
                        0,
                        usize::MAX,
                    )),
                },
                (Precomputed(v), Infinite(b)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(Fixed(v)), Box::new(b)))
                    }
                    Operator::Multiply | Operator::Divide => Precomputed(generator.generate(
                        &BinaryPointOp(op, Box::new(Fixed(v)), Box::new(b)),
                        0,
                        usize::MAX,
                    )),
                },
                // At least one is Dynamic
                (a, b) => Dynamic(BinaryPointOp(op, Box::new(a.into()), Box::new(b.into()))),
            },
            Filter {
                waveform,
                feed_forward,
                feedback,
                state,
            } => {
                match (
                    precompute_internal(generator, *waveform),
                    precompute_internal(generator, *feed_forward),
                    precompute_internal(generator, *feedback),
                ) {
                    (Precomputed(w), Precomputed(ff), Precomputed(fb)) => {
                        let ff_len = ff.len();
                        Precomputed(generator.generate(
                            &Filter {
                                waveform: Box::new(Fixed(w)),
                                feed_forward: Box::new(Fixed(ff)),
                                feedback: Box::new(Fixed(fb)),
                                state,
                            },
                            0,
                            usize::MAX - (ff_len - 1), // because Filter will add to desired
                        ))
                    }
                    (Dynamic(waveform), feed_forward, feedback) => Dynamic(Filter {
                        waveform: Box::new(waveform),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                    (waveform, Dynamic(feed_forward), feedback) => Dynamic(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                    (waveform, feed_forward, Dynamic(feedback)) => Dynamic(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback),
                        state,
                    }),
                    // None are Dynamic, at least one is Infinite
                    (waveform, feed_forward, feedback) => Infinite(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                }
            }
            Res {
                trigger,
                waveform,
                state,
            } => {
                match (
                    precompute_internal(generator, *trigger),
                    precompute_internal(generator, *waveform),
                ) {
                    (Precomputed(t), Precomputed(w)) => Precomputed(generator.generate(
                        &Res {
                            trigger: Box::new(Fixed(t)),
                            waveform: Box::new(Fixed(w)),
                            state,
                        },
                        0,
                        usize::MAX,
                    )),
                    (Dynamic(trigger), waveform) => Dynamic(Res {
                        trigger: Box::new(trigger),
                        waveform: Box::new(waveform.into()),
                        state,
                    }),
                    (trigger, Dynamic(waveform)) => Dynamic(Res {
                        trigger: Box::new(trigger.into()),
                        waveform: Box::new(waveform),
                        state,
                    }),
                    // Neither is Dynamic, at least one is Infinite
                    (trigger, waveform) => Infinite(Res {
                        trigger: Box::new(trigger.into()),
                        waveform: Box::new(waveform.into()),
                        state,
                    }),
                }
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                match (
                    precompute_internal(generator, *trigger),
                    precompute_internal(generator, *positive_waveform),
                    precompute_internal(generator, *negative_waveform),
                ) {
                    (Precomputed(t), Precomputed(p), Precomputed(n)) => {
                        Precomputed(generator.generate(
                            &Alt {
                                trigger: Box::new(Fixed(t)),
                                positive_waveform: Box::new(Fixed(p)),
                                negative_waveform: Box::new(Fixed(n)),
                            },
                            0,
                            usize::MAX,
                        ))
                    }
                    (Dynamic(waveform), positive_waveform, negative_waveform) => Dynamic(Alt {
                        trigger: Box::new(waveform),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                    (waveform, Dynamic(positive_waveform), negative_waveform) => Dynamic(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                    (waveform, positive_waveform, Dynamic(negative_waveform)) => Dynamic(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform),
                    }),
                    // None are Dynamic, at least one is Infinite
                    (waveform, positive_waveform, negative_waveform) => Infinite(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                }
            }
            Slider(_) | Marked { .. } | Captured { .. } => Dynamic(waveform),
        }
    }

    return precompute_internal(generator, waveform).into();
}

#[cfg(test)]
mod tests {
    use std::f32;

    use super::*;
    use crate::optimizer;
    use waveform::Operator;
    use waveform::Waveform;
    use waveform::Waveform::{
        Append, BinaryPointOp, Const, Filter, Fin, Fixed, Res, Seq, Sin, Time,
    };

    const MAX_LENGTH: usize = 1000;

    impl<T> MaybeOption<T> {
        fn is_some(&self) -> bool {
            matches!(self, MaybeOption::Some(_))
        }

        fn unwrap(self) -> T {
            match self {
                MaybeOption::Some(v) => v,
                _ => panic!("Called unwrap on a MaybeOption that wasn't Some"),
            }
        }
    }

    fn new_test_generator<'a>(sample_frequency: i32) -> Generator<'a> {
        Generator::new(sample_frequency)
    }

    fn finite_const_waveform(value: f32, fin_duration: u64, seq_duration: u64) -> Waveform {
        return Seq {
            offset: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time),
                Box::new(Const(seq_duration as f32)),
            )),
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(fin_duration as f32)),
                )),
                waveform: Box::new(Const(value)),
            }),
        };
    }

    fn run_tests(waveform: &waveform::Waveform, desired: Vec<f32>) {
        let length =
            new_test_generator(1).remaining(&initialize_state(waveform.clone()), 0, desired.len());
        for size in [1, 2, 4, 8] {
            let generator = new_test_generator(1);
            let w = initialize_state(waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = generator.generate(&w, (n * size) as i64, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, desired,
                "Failed output for size {} on waveform:\n{:#?}",
                size, waveform
            );
        }

        for size in [1, 2, 4, 8] {
            let generator = new_test_generator(1);
            let (_, optimized_waveform) = optimizer::replace_seq(waveform.clone());
            let w = initialize_state(optimized_waveform.clone());
            assert_eq!(
                length,
                generator.remaining(&w, 0, desired.len()),
                "Failed length on waveform\n{:#?}\nwith seq's removed\n{:#?}",
                waveform,
                optimized_waveform
            );
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = generator.generate(&w, (n * size) as i64, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, desired,
                "Failed output for size {} on waveform\n{:#?}\nwith seq's removed\n{:#?}",
                size, waveform, optimized_waveform
            );
        }

        for size in [1, 2, 4, 8] {
            let generator = new_test_generator(1);
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            let optimized_waveform = optimizer::simplify(no_seq_waveform.clone());
            let w = initialize_state(optimized_waveform.clone());
            assert_eq!(
                length,
                generator.remaining(&w, 0, desired.len()),
                "Failed length on waveform\n{:#?}\nwith seq's removed\n{:#?}\noptimized to\n{:#?}",
                waveform,
                no_seq_waveform,
                optimized_waveform
            );
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = generator.generate(&w, (n * size) as i64, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, desired,
                "Failed output for size {} on waveform\n{:#?}\noptimized to\n{:#?}",
                size, waveform, optimized_waveform
            );
        }

        /*
        for size in [1, 2, 4, 8] {
            let generator = new_test_generator(1);
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            let optimized_waveform = optimizer::simplify(no_seq_waveform.clone());
            let w = optimizer::precompute(&generator, initialize_state(optimized_waveform.clone()));
            assert_eq!(
                length,
                generator.remaining(&w, 0, desired.len()),
                "Failed length on waveform\n{:#?}\nprecomputed to\n{:#?}",
                waveform,
                remove_state(w.clone()),
            );
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = generator.generate(&w, (n * size) as i64, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, desired,
                "Failed output for size {} on waveform\n{:#?}\nprecomputed to\n{:#?}",
                size, waveform, remove_state(w)
            );
        }
        */
    }

    #[test]
    fn test_time() {
        let w1 = Waveform::Time;
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let generator = new_test_generator(1);
        let result = generator.generate(&initialize_state(w1), 0, 8);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    // frequency in Hz, phase in radians
    fn sin_waveform(frequency: f32, phase: f32) -> Box<Waveform> {
        return Box::new(Sin {
            frequency: Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(f32::consts::TAU)),
                Box::new(Const(frequency)),
            )),
            phase: Box::new(Const(phase)),
            state: (),
        });
    }

    fn run_sin_test(
        generator: &Generator,
        waveform: &super::Waveform,
        expected: Vec<f32>,
        position: i64,
    ) {
        let result = generator.generate(waveform, position, expected.len());
        assert_ne!(
            result[0], expected[0],
            "First sample shouldn't match exactly"
        );
        for (i, &x) in result.iter().enumerate() {
            assert!(
                x - expected[i] < 1e-4,
                "result = {}, expected = {}, result - expected = {}",
                x,
                expected[i],
                x - expected[i]
            );
        }
    }

    #[test]
    fn test_sin() {
        let sample_frequency = 100.0;
        let generator = new_test_generator(sample_frequency as i32);

        let w1 = initialize_state(*sin_waveform(1.0, 0.0));
        let expected = generator.generate(&w1, 0, 8);
        // Now generate from the same point in the sine wave at a much greater position
        generator.generate(&w1, 8, 9992);
        run_sin_test(&generator, &w1, expected, 10000);

        let w2 = initialize_state(*sin_waveform(1.0, 0.0));
        generator.generate(&w2, 0, 3);
        let expected = generator.generate(&w2, 3, 8);
        // Same but from the middle of the sine wave.
        generator.generate(&w2, 11, 9992);
        run_sin_test(&generator, &w2, expected, 10003);

        let w3 = initialize_state(*sin_waveform(3.0, 0.0));
        let expected = generator.generate(&w3, 0, 8);
        generator.generate(&w3, 8, 92);
        run_sin_test(&generator, &w3, expected, 100);

        // Non-constant frequency
        let w4 = initialize_state(Waveform::Sin {
            frequency: Box::new(Time),
            phase: Box::new(Const(0.0)),
            state: (),
        });
        // XXX this is the naive and incorrect indirect method so it shouldn't work
        generator.generate(&w4, 0, 20);
        let expected = (20..120)
            .map(|sample| {
                let t = sample as f64 / sample_frequency;
                (t * t).sin() as f32
            })
            .collect();
        run_sin_test(&generator, &w4, expected, 20);
    }

    #[test]
    fn test_res() {
        let w1 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time),
            state: (),
        };
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w2 = Res {
            trigger: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(6.0)),
                )),
                waveform: sin_waveform(0.25, 0.0),
            }),
            waveform: Box::new(Time),
            state: (),
        };
        let generator = new_test_generator(1);
        assert_eq!(
            generator.remaining(&initialize_state(w2.clone()), 0, MAX_LENGTH),
            6
        );
        run_tests(&w2, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0]);

        let w3 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Waveform::Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(3.0)),
                )),
                waveform: Box::new(Waveform::Time),
            }),
            state: (),
        };
        run_tests(&w3, vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);

        // Test a reset that occurs before time 0
        let w4 = Res {
            trigger: Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Time),
                Box::new(Const(2.0)),
            )),
            waveform: Box::new(Time),
            state: (),
        };
        run_tests(&w4, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let w5 = Res {
            trigger: sin_waveform(0.25, f32::consts::PI),
            waveform: Box::new(Time),
            state: (),
        };
        run_tests(&w5, vec![2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

        // Test where a reset lines up with the buffer boundary and where there are multiple
        // resets in a buffer.
        let w6 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time),
            state: (),
        };
        run_tests(
            &w6,
            vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
            ],
        );
    }

    #[test]
    fn test_append() {
        let generator = new_test_generator(1);
        let w1 = Append(
            Box::new(finite_const_waveform(1.0, 3, 2)),
            Box::new(finite_const_waveform(2.0, 3, 2)),
        );
        assert_eq!(
            generator.offset(&initialize_state(w1.clone()), MAX_LENGTH),
            4
        );
        assert_eq!(
            generator.remaining(&initialize_state(w1.clone()), 0, MAX_LENGTH),
            6
        );
        run_tests(&w1, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sum() {
        let generator = new_test_generator(1);
        let w1 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(1.0, 5, 2)),
            Box::new(finite_const_waveform(1.0, 5, 2)),
        );
        assert_eq!(
            generator.offset(&initialize_state(w1.clone()), MAX_LENGTH),
            4
        );
        assert_eq!(
            generator.remaining(&initialize_state(w1.clone()), 0, MAX_LENGTH),
            7
        );
        run_tests(&w1, vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0]);

        let w2 = Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time),
                Box::new(Const(8.0)),
            )),
            waveform: Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Seq {
                    offset: Box::new(Const(0.0)),
                    waveform: Box::new(Const(1.0)),
                }),
                Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Seq {
                        offset: Box::new(Const(0.0)),
                        waveform: Box::new(Const(2.0)),
                    }),
                    Box::new(Fin {
                        length: Box::new(Const(0.0)),
                        waveform: Box::new(Const(0.0)),
                    }),
                )),
            )),
        };
        run_tests(&w2, vec![3.0; 8]);

        let w5 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 2, 2)),
        );
        run_tests(&w5, vec![3.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0]);

        // Test a case to make sure that the sum generates enough samples, even when
        // the left-hand side is shorter and the right hasn't started yet.
        let result = generator.generate(&initialize_state(w5.clone()), 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);
        let result = generator.generate(&initialize_state(w5), 1, 2);
        assert_eq!(result, vec![0.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 0, 0)),
        );
        let result = generator.generate(&initialize_state(w6), 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);

        let w7 = BinaryPointOp(
            Operator::Add,
            Box::new(Fixed(vec![1.0])),
            Box::new(Const(0.0)),
        );
        run_tests(&w7, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let generator = new_test_generator(1);
        let w1 = BinaryPointOp(
            Operator::Multiply,
            Box::new(finite_const_waveform(3.0, 8, 2)),
            Box::new(finite_const_waveform(2.0, 5, 2)),
        );
        assert_eq!(
            generator.offset(&initialize_state(w1.clone()), MAX_LENGTH),
            4
        );
        assert_eq!(
            generator.remaining(&initialize_state(w1.clone()), 0, MAX_LENGTH),
            7
        );
        run_tests(&w1, vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0]);

        let w2 = BinaryPointOp(
            Operator::Multiply,
            Box::new(finite_const_waveform(3.0, 5, 2)),
            Box::new(finite_const_waveform(2.0, 5, 2)),
        );
        run_tests(&w2, vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]);

        let w3 = Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time),
                Box::new(Const(8.0)),
            )),
            waveform: Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(3.0)),
                Box::new(Const(2.0)),
            )),
        };
        run_tests(&w3, vec![6.0; 8]);

        let w4 = BinaryPointOp(
            Operator::Multiply,
            Box::new(Seq {
                offset: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(1.0)),
                )),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_waveform(2.0, 5, 5)),
        );
        run_tests(&w4, vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_filter() {
        // FIRs
        let w1 = Filter {
            waveform: Box::new(Time),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0])),
            feedback: Box::new(Fixed(vec![])),
            state: (),
        };
        run_tests(&w1, vec![-6.0, 0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0]);

        let w2 = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(5.0)),
                )),
                waveform: Box::new(Time),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0])),
            feedback: Box::new(Fixed(vec![])),
            state: (),
        };
        let generator = new_test_generator(1);
        assert_eq!(
            generator.remaining(&initialize_state(w2.clone()), 0, MAX_LENGTH),
            5
        );
        run_tests(&w2, vec![-6.0, 0.0, 6.0, 12.0, 18.0, 0.0, 0.0, 0.0]);

        let w3 = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time),
                    Box::new(Const(3.0)),
                )),
                waveform: Box::new(Time),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0, 2.0, 2.0])),
            feedback: Box::new(Fixed(vec![])),
            state: (),
        };
        let generator = new_test_generator(1);
        assert_eq!(
            generator.remaining(&initialize_state(w3.clone()), 0, MAX_LENGTH),
            3
        );
        run_tests(&w3, vec![-20.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let w4 = Filter {
            waveform: Box::new(Res {
                // Pick a trigger that's far from zero on at our sampled points
                trigger: sin_waveform(1.0 / 3.0, 3.0 * std::f32::consts::PI / 2.0),
                waveform: Box::new(Time),
                state: (),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0])),
            feedback: Box::new(Fixed(vec![])),
            state: (),
        };
        run_tests(&w4, vec![6.0, 4.0, 2.0, 6.0, 4.0, 2.0, 6.0, 4.0]);

        let w5 = Filter {
            waveform: Box::new(Const(1.0)),
            feed_forward: Box::new(Fixed(vec![0.2, 0.2, 0.2, 0.2, 0.2])),
            feedback: Box::new(Fixed(vec![])),
            state: (),
        };
        run_tests(&w5, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // IIRs
        let w6 = Filter {
            waveform: Box::new(Time),
            feed_forward: Box::new(Fixed(vec![0.5])),
            feedback: Box::new(Fixed(vec![-0.5])),
            state: (),
        };
        run_tests(
            &w6,
            vec![0.0, 0.5, 1.25, 2.125, 3.0625, 4.03125, 5.015625, 6.0078125],
        );

        // Cascade
        let w7 = Filter {
            waveform: Box::new(Filter {
                waveform: Box::new(Time),
                feed_forward: Box::new(Fixed(vec![0.5])),
                feedback: Box::new(Fixed(vec![-0.5])),
                state: (),
            }),
            feed_forward: Box::new(Fixed(vec![0.4])),
            feedback: Box::new(Fixed(vec![-0.6])),
            state: (),
        };
        run_tests(
            &w7,
            vec![
                0.0, 0.2, 0.62, 1.222, 1.9582, 2.7874203, 3.6787024, 4.610347,
            ],
        );
    }

    #[test]
    fn test_greater_equals_at() {
        let w1 = BinaryPointOp(Operator::Add, Box::new(Time), Box::new(Const(-5.0)));
        let w2 = Fin {
            length: Box::new(w1.clone()),
            waveform: Box::new(Time),
        };
        let generator = new_test_generator(1);
        let out = generator.generate(&initialize_state(w2), 0, 10);
        let position = generator.greater_or_equals_at(&initialize_state(w1), 0.0, 0, 10);
        assert!(position.is_some());
        assert_eq!(position.unwrap(), out.len());
        for (i, x) in out.iter().enumerate() {
            if i < position.unwrap() {
                assert_eq!(*x, i as f32);
            } else if i == position.unwrap() {
                assert!(*x >= 0.0);
            }
        }
    }

    // TODO test for forgetting to sort pending waveforms
}
