use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::BufWriter;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use fastrand;

extern crate sdl2;
use sdl2::audio::AudioCallback;

// TODO move this out of the tracker?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slider {
    X,
    Y,
}

impl fmt::Display for Slider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Slider::X => write!(f, "X"),
            Slider::Y => write!(f, "Y"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Waveform<State = ()> {
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
     * Fin generates a finite waveform, truncating the underlying waveform. The length is determined
     * by the first point at which the `length` waveform is >= 0.0. For example, `Fin(Const(0.0), _)`
     * is 0 seconds in length and `Fin(Sum(Time, Const(-2.0)), _)` is 2 seconds in length.
     */
    Fin {
        length: Box<Waveform<State>>,
        waveform: Box<Waveform<State>>,
    },
    /*
     * Seq sets the offset of `waveform`. The offset is determined by the first point at which the `offset` waveform
     * is >= 0.0. For example, `Seq(Const(0.0), _)` is has an offset 0 seconds and `Seq(Sum(Time, Const(-2.0)))` has
     * an offset of 2 seconds.
     */
    Seq {
        offset: Box<Waveform<State>>,
        waveform: Box<Waveform<State>>,
    },
    /*
     * Append concatenates two waveforms, generating all samples from the first waveform and
     * then all samples from the second (regardless of the offset of the first).
     */
    Append(Box<Waveform<State>>, Box<Waveform<State>>),
    /*
     * Sin computes the sine of each sample in the given waveform.
     */
    Sin(Box<Waveform<State>>),
    /*
     * Filter implements an impulse response filter with feed-forward and feedback coefficients. Assumes that the first
     * feedback coefficient (a_0) is 1.0. If the filter has no feedback coefficients, then the filter has a finite
     * response -- that is, it is a convolution.
     */
    // TODO maybe add a_0 back in?
    Filter {
        waveform: Box<Waveform<State>>,
        feed_forward: Box<Waveform<State>>, // b_0, b_1, ...
        feedback: Box<Waveform<State>>,     // a_1, a_2, ...
        state: State,
    },
    Sum(Box<Waveform<State>>, Box<Waveform<State>>),
    DotProduct(Box<Waveform<State>>, Box<Waveform<State>>),
    /*
     * Res generates a repeating waveform that restarts the given waveform whenever the trigger
     * waveform flips from negative values to positive values. Its length and offset are determined
     * by the trigger waveform.
     */
    Res {
        trigger: Box<Waveform<State>>,
        waveform: Box<Waveform<State>>,
    },
    /*
     * Alt generates a waveform by alternating between two waveforms based on the sign of
     * the trigger waveform.
     */
    Alt {
        trigger: Box<Waveform<State>>,
        positive_waveform: Box<Waveform<State>>,
        negative_waveform: Box<Waveform<State>>,
    },
    /*
     * Slider generates samples from an interactive "slider" input.
     */
    Slider(Slider),
    /*
     * Marked waveforms generate the same samples as the inner waveform and are used to signal that a certain
     * event will occur or has occurred. Each status update will include a list of marked waveforms, along with
     * their start times and durations.
     */
    Marked {
        id: u32,
        waveform: Box<Waveform<State>>,
    },
    /* Captured waveforms generate the same samples as the inner waveform and also write them to a file
     * beginning with the given file stem.
     */
    Captured {
        file_stem: String,
        waveform: Box<Waveform<State>>,
    },
}

impl fmt::Display for Waveform<()> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Waveform::*;
        match self {
            Const(value) => write!(f, "Const({})", value),
            Time => write!(f, "Time"),
            Noise => write!(f, "Noise"),
            Fixed(samples) => write!(f, "Fixed({:?})", samples),
            Fin { length, waveform } => {
                write!(f, "Fin({}, {})", length, waveform)
            }
            Seq { offset, waveform } => {
                write!(f, "Seq({}, {})", offset, waveform)
            }
            Append(a, b) => write!(f, "Append({}, {})", a, b),
            Sin(waveform) => write!(f, "Sin({})", waveform),
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => write!(f, "Filter({}, {}, {})", waveform, feed_forward, feedback),
            Sum(a, b) => write!(f, "Sum({}, {})", a, b),
            DotProduct(a, b) => write!(f, "DotProduct({}, {})", a, b),
            Res { trigger, waveform } => {
                write!(f, "Res({}, {})", trigger, waveform)
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => write!(
                f,
                "Alt({}, {}, {})",
                trigger, positive_waveform, negative_waveform
            ),
            Slider(slider) => write!(f, "Slider({})", slider),
            Marked { id, waveform } => {
                write!(f, "Marked({}, {})", id, waveform)
            }
            Captured {
                file_stem,
                waveform,
            } => {
                write!(f, "Captured({}, {})", file_stem, waveform)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilterState {
    // The state of the filter, used to store previously generated samples and the position of the first of those
    // samples. Used as an input to feedback.
    previous_out: RefCell<(Vec<f32>, i64)>,
}

struct SliderState {
    // The final values of sliders at the end of the last generate call
    last_values: HashMap<Slider, f32>,
    // Any changes to the sliders since the last generate call
    changes: HashMap<Slider, f32>,
    // The length of the current buffer being generated
    buffer_length: usize,
    // The position in the buffer where the next sample will be written
    buffer_position: usize,
}

/*
 * Generator converts waveforms into sequences of samples.
 */
pub struct Generator<'a> {
    sample_frequency: i32,
    slider_state: Option<&'a SliderState>,
    capture_state:
        Option<RefCell<&'a mut HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>>>,
}

// TODO add metrics for waveform expr depth and total ops

#[derive(Debug, Copy, Clone)]
enum MaybeOption<T> {
    Some(T),
    None,
    Maybe, // The value may or may not be present
}

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
    pub fn generate(
        &self,
        waveform: &Waveform<FilterState>,
        position: i64,
        desired: usize,
    ) -> Vec<f32> {
        use Waveform::*;
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
                    // We have remaining for a == 0 at position, and we need to figure out how far into b we are.
                    // Get the length of a, and use that to compute the position into b. If remaining is 0 then
                    // position can't be negative (since no waveform can end before position 0).
                    assert!(position >= 0);
                    let length = self.remaining(a, 0, desired + position as usize);
                    return self.generate(b, position - length as i64, desired);
                }
            }
            Sin(waveform) => {
                let mut out = self.generate(waveform, position, desired);
                for f in out.iter_mut() {
                    *f = (*f).sin();
                }
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
                const MAX_FILTER_LENGTH: usize = 1000; // Maximum length of the filter coefficients
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
            Sum(a, b) => {
                return self.generate_binary_op(|x, y| x + y, a, b, position, desired);
            }
            DotProduct(a, b) => {
                // Like sum, but we need to make sure we generate a length based on
                // the shorter waveform.
                let remaining = self.remaining(waveform, position, desired);
                return self.generate_binary_op(|x, y| x * y, a, b, position, remaining);
            }
            Res { trigger, waveform } => {
                // Maybe cache the last trigger position and signum and use it if position doesn't change? So we don't need this constant...
                let max_reset_lookback: i64 = 10000; // Maximum number of samples to look back for a reset trigger
                // TODO think about all of these unwrap_ors
                // TODO generate the trigger in blocks?
                // TODO think about the interaction of this non-monotone position with Filter

                // First go back and find the most recent trigger before position.
                let mut last_trigger_position = position;
                let trigger_out = self.generate(trigger, position, 1);
                if trigger_out.is_empty() {
                    return vec![];
                }
                let mut last_signum = trigger_out[0].signum();
                loop {
                    if position - last_trigger_position > max_reset_lookback {
                        panic!(
                            "No reset trigger found within {} samples before position {} in waveform {:?}",
                            max_reset_lookback, position, trigger
                        );
                    }
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
                    // Set to true if a restart will be triggered before desired is reached
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
                        inner_position += inner_desired as i64;
                    }
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
                    .unwrap_or(0.0);
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
                                .max(-1.0);
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
                if self.capture_state.is_none() {
                    println!("Warning: captured waveform used without capture_state");
                    return vec![0.0; desired];
                }
                let out = self.generate(waveform, position, desired);
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

    // Generate a binary operation on two waveforms, up to 'desired' samples starting at 'position'
    // relative to the start of the first waveform. The second waveform is offset by the
    // offset of the first waveform. The `op` function is applied to each pair of samples.
    fn generate_binary_op(
        &self,
        op: fn(f32, f32) -> f32,
        a: &Waveform<FilterState>,
        b: &Waveform<FilterState>,
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

        // Calculate the maximum offset of a that is relevant for position and desired. This as the sum
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
    fn remaining(&self, waveform: &Waveform<FilterState>, position: i64, max: usize) -> usize {
        use Waveform::*;
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
            Seq { waveform, .. } | Sin(waveform) | Filter { waveform, .. } => {
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
            Sum(a, b) | DotProduct(a, b) => {
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
                match waveform {
                    Sum(_, _) => from_a.max(from_b),
                    DotProduct(_, _) => from_a.min(from_b),
                    _ => unreachable!(),
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
        waveform: &Waveform<FilterState>,
        value: f32,
        position: i64,
        max: usize,
    ) -> MaybeOption<usize> {
        use Waveform::{Append, Const, Sum, Time};
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
            Sum(a, b) => {
                // If a has an offset, we'd need to handle the overlapping and non-overlapping parts
                // separately, and since we don't expect this to be called on non-optimized waveforms (except
                // in tests), just give up.
                if self.offset(a, max) != 0 {
                    return MaybeOption::Maybe;
                }
                match (&**a, &**b) {
                    (Const(va), Const(vb)) => {
                        if va + vb >= value {
                            MaybeOption::Some(0)
                        } else {
                            MaybeOption::None
                        }
                    }
                    (Const(va), _) => self.greater_or_equals_at(b, value - va, position, max),
                    (_, Const(vb)) => self.greater_or_equals_at(a, value - vb, position, max),
                    _ => MaybeOption::Maybe,
                }
            }
            _ => MaybeOption::Maybe,
        }
    }

    // Returns the offset at which the next waveform should start (in samples). This is determined entirely by
    // `waveform` (this function is pure).
    fn offset(&self, waveform: &Waveform<FilterState>, max: usize) -> usize {
        match waveform {
            Waveform::Const { .. } | Waveform::Time | Waveform::Noise | Waveform::Fixed(_) => 0,
            Waveform::Fin { waveform, .. } => self.offset(waveform, max),
            Waveform::Seq { offset, .. } => match self.greater_or_equals_at(&offset, 0.0, 0, max) {
                MaybeOption::Some(new_position) => new_position,
                MaybeOption::None => max,
                MaybeOption::Maybe => {
                    panic!(
                        "Unable to determine offset of Seq offset cheaply: {:?}",
                        offset
                    );
                }
            },
            Waveform::Append(a, b) => (self.offset(a, max) + self.offset(b, max)).min(max),
            Waveform::Sin(waveform) => self.offset(waveform, max),
            Waveform::Filter { waveform, .. } => self.offset(waveform, max),
            Waveform::Sum(a, b) | Waveform::DotProduct(a, b) => {
                let a_offset = self.offset(a, max);
                if a_offset == max {
                    return max;
                }
                let b_offset = self.offset(b, max - a_offset);
                return a_offset + b_offset;
            }
            Waveform::Res { trigger, .. } | Waveform::Alt { trigger, .. } => {
                self.offset(trigger, max)
            }
            Waveform::Slider { .. } => 0,
            Waveform::Marked { waveform, .. } | Waveform::Captured { waveform, .. } => {
                self.offset(waveform, max)
            }
        }
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
    MoveSlider {
        // The slider to set
        slider: Slider,
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
    pub slider_values: HashMap<Slider, f32>,
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
    waveform: Waveform<FilterState>,
    marks: Vec<Mark<I>>,
    position: usize,
    // Open files used by Captured waveforms
    capture_state: HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
}

#[derive(Debug, Clone)]
struct PendingWaveform<I> {
    id: I,
    waveform: Waveform,
    start: Instant,
    repeat_every: Option<Duration>,
    marks: Vec<Mark<I>>,
}

pub fn initialize_state(waveform: Waveform) -> Waveform<FilterState> {
    use Waveform::*;
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
        Sin(waveform) => Sin(Box::new(initialize_state(*waveform))),
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
        Sum(a, b) => Sum(
            Box::new(initialize_state(*a)),
            Box::new(initialize_state(*b)),
        ),
        DotProduct(a, b) => DotProduct(
            Box::new(initialize_state(*a)),
            Box::new(initialize_state(*b)),
        ),
        Res { trigger, waveform } => Res {
            trigger: Box::new(initialize_state(*trigger)),
            waveform: Box::new(initialize_state(*waveform)),
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

pub struct Tracker<I>
where
    I: Clone + Send,
{
    sample_frequency: i32,
    command_receiver: mpsc::Receiver<Command<I>>,
    status_sender: mpsc::Sender<Status<I>>,

    // Persistent generation state
    active_waveforms: Vec<ActiveWaveform<I>>,
    pending_waveforms: Vec<PendingWaveform<I>>, // sorted by start time
    // Command state
    send_current_buffer: bool,
    slider_state: SliderState,
}

impl<I> Tracker<I>
where
    I: Clone + Send,
{
    pub fn new(
        sample_frequency: i32,
        command_receiver: mpsc::Receiver<Command<I>>,
        status_sender: mpsc::Sender<Status<I>>,
    ) -> Tracker<I> {
        return Tracker {
            sample_frequency,
            command_receiver,
            status_sender,

            active_waveforms: Vec::new(),
            pending_waveforms: Vec::new(),

            send_current_buffer: false,
            slider_state: SliderState {
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
        waveform: &Waveform<FilterState>,
        out: &mut Vec<Mark<I>>,
    ) {
        use Waveform::*;
        match waveform {
            Const(_) | Time | Noise | Fixed(_) | Slider { .. } => {
                return;
            }
            // TODO Fin seems not quite right here, since its length might truncate any marks inside it
            Fin { waveform, .. }
            | Seq { waveform, .. }
            | Sin(waveform)
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
            Append(a, b) => {
                self.process_marked(waveform_id, start, &*a, out);
                let remaining = Generator::new(self.sample_frequency).remaining(
                    &*a,
                    0,
                    10 * self.sample_frequency as usize, // XXX
                );
                let start = start
                    + Duration::from_secs_f32(remaining as f32 / self.sample_frequency as f32);
                self.process_marked(waveform_id, start, &*b, out);
            }
            Sum(a, b) | DotProduct(a, b) => {
                self.process_marked(waveform_id, start, &*a, out);
                // TODO do something about this constant for max
                let offset = Generator::new(self.sample_frequency)
                    .offset(&*a, 10 * self.sample_frequency as usize);
                let start =
                    start + Duration::from_secs_f32(offset as f32 / self.sample_frequency as f32);
                self.process_marked(waveform_id, start, &*b, out);
            }
            Marked { waveform, id } => {
                let length = Generator::new(self.sample_frequency).remaining(
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

    fn process_captured(
        &self,
        waveform: &Waveform,
        out: &mut HashMap<String, hound::WavWriter<BufWriter<std::fs::File>>>,
    ) {
        use Waveform::*;
        match waveform {
            Const(_) | Time | Noise | Fixed(_) | Slider { .. } => {
                return;
            }
            Fin { waveform, .. }
            | Seq { waveform, .. }
            | Sin(waveform)
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
            Append(a, b) | Sum(a, b) | DotProduct(a, b) => {
                self.process_captured(&*a, out);
                self.process_captured(&*b, out);
            }
            Captured {
                file_stem,
                waveform,
            } => {
                use hound::{SampleFormat, WavSpec, WavWriter};
                self.process_captured(&*waveform, out);
                // If we haven't already opened a file for this waveform, then open it now.
                if out.contains_key(file_stem) {
                    panic!(
                        "Captured waveform called with duplicate file stem: {}",
                        file_stem
                    );
                }
                let datetime = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
                let file_name = format!("{}_{}.wav", &file_stem, &datetime);
                let path = std::path::Path::new(&file_name);
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
            let last_value = self.slider_state.last_values.remove(slider).unwrap_or(0.0);
            self.slider_state
                .last_values
                .insert(slider.clone(), (last_value + change).min(1.0).max(-1.0));
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
                self.process_marked(&id, start, &initialize_state(waveform.clone()), &mut marks);
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
                // TODO This is wrong in the case where a long time has passed.
                // For example, it's ok to adjust the start to segment_start if the difference is small,
                // but if it's greater than repeat_every, then we can end up aligning the two Beats
                // waveforms with each other (instead of interleaving them).
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
                    let mut capture_state = HashMap::new();
                    self.process_captured(&pending.waveform, &mut capture_state);
                    self.active_waveforms.push(ActiveWaveform {
                        id: pending.id.clone(),
                        waveform: initialize_state(pending.waveform.clone()),
                        marks,
                        position: 0,
                        capture_state,
                    });
                    // Check to see if this waveform should repeat
                    if let Some(duration) = pending.repeat_every {
                        println!(
                            "Scheduling waveform {:?} to repeat after {:?} (at {:?})",
                            pending.id,
                            duration,
                            segment_start + duration
                        );
                        pending.start = segment_start + duration;
                        pending.marks = Vec::new();
                        self.process_marked(
                            &pending.id,
                            pending.start,
                            &initialize_state(pending.waveform.clone()),
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
                    let mut generator = Generator::new(self.sample_frequency);
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
                    println!(
                        "Removing waveform {:?} at position {} and time {:?}",
                        active.id,
                        active.position,
                        segment_start
                            + Duration::from_secs_f32(
                                tmp.len() as f32 / self.sample_frequency as f32
                            )
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
        }
        return finished;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer;
    use Waveform::{Append, Const, DotProduct, Filter, Fin, Fixed, Res, Seq, Sin, Sum, Time};

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
            offset: Box::new(Sum(Box::new(Time), Box::new(Const(-(seq_duration as f32))))),
            waveform: Box::new(Fin {
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-(fin_duration as f32))))),
                waveform: Box::new(Const(value)),
            }),
        };
    }

    fn run_tests(waveform: &Waveform, desired: Vec<f32>) {
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
    }

    #[test]
    fn test_time() {
        let w1 = Waveform::Time;
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let generator = new_test_generator(1);
        let result = generator.generate(&initialize_state(w1), 0, 8);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    fn sin_waveform(frequency: f32, phase: f32) -> Box<Waveform> {
        return Box::new(Sin(Box::new(Sum(
            Box::new(DotProduct(
                Box::new(Const(2.0)),
                Box::new(DotProduct(
                    Box::new(Const(std::f32::consts::PI)),
                    Box::new(DotProduct(Box::new(Const(frequency)), Box::new(Time))),
                )),
            )),
            Box::new(Const(phase)),
        ))));
    }

    #[test]
    fn test_res() {
        let w1 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time),
        };
        run_tests(&w1, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w2 = Res {
            trigger: Box::new(Fin {
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-6.0)))),
                waveform: sin_waveform(0.25, 0.0),
            }),
            waveform: Box::new(Time),
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
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-3.0)))),
                waveform: Box::new(Waveform::Time),
            }),
        };
        run_tests(&w3, vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);

        // Test a reset that occurs before time 0
        let w4 = Res {
            trigger: Box::new(Sum(Box::new(Time), Box::new(Const(2.0)))),
            waveform: Box::new(Time),
        };
        run_tests(&w4, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
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
        let w1 = Sum(
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
            length: Box::new(Sum(Box::new(Time), Box::new(Const(-8.0)))),
            waveform: Box::new(Sum(
                Box::new(Seq {
                    offset: Box::new(Const(0.0)),
                    waveform: Box::new(Const(1.0)),
                }),
                Box::new(Sum(
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

        let w5 = Sum(
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
        let w6 = Sum(
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 0, 0)),
        );
        let result = generator.generate(&initialize_state(w6), 0, 2);
        assert_eq!(result, vec![3.0, 0.0]);

        let w7 = Sum(Box::new(Fixed(vec![1.0])), Box::new(Const(0.0)));
        run_tests(&w7, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let generator = new_test_generator(1);
        let w1 = DotProduct(
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

        let w2 = DotProduct(
            Box::new(finite_const_waveform(3.0, 5, 2)),
            Box::new(finite_const_waveform(2.0, 5, 2)),
        );
        run_tests(&w2, vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]);

        let w3 = Fin {
            length: Box::new(Sum(Box::new(Time), Box::new(Const(-8.0)))),
            waveform: Box::new(DotProduct(Box::new(Const(3.0)), Box::new(Const(2.0)))),
        };
        run_tests(&w3, vec![6.0; 8]);

        let w4 = DotProduct(
            Box::new(Seq {
                offset: Box::new(Sum(Box::new(Time), Box::new(Const(-1.0)))),
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
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-5.0)))),
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
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-3.0)))),
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
        let w1 = Sum(Box::new(Time), Box::new(Const(-5.0)));
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
