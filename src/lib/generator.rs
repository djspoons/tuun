use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::io::BufWriter;
use std::{f64, mem, usize};

use fastrand;

use crate::waveform;

#[derive(Debug, Clone)]
pub enum State {
    // The state of a waveform that hasn't yet generated any samples.
    Initial,
    // The position in the current waveform as the number of samples since the beginning of the waveform.
    Position(usize),
    // TODO consider a shared vec plus a slice for Fixed? how to represent a negative position? and direction?
    // Shared(Rc<Vec<f32>>, &[f32]),
    // Previously generated samples as input and/or output to the waveform (for example, for Filter)
    Samples { input: Vec<f32>, output: Vec<f32> },
    // The current accumulated phase (for example, for Sine).
    Phase { accumulator: f64 },
    // The sign of the last generated value along with the direction of generation (for example, for Reset).
    Sign { signum: f32 },
}

pub type Waveform<M> = waveform::Waveform<M, State>;

pub fn initialize_state<M, S>(waveform: waveform::Waveform<M, S>) -> Waveform<M> {
    return waveform::initialize_state(waveform, State::Initial);
}

/*
 * Generator converts waveforms into sequences of samples.
 */
pub struct Generator<'a> {
    sample_rate: i32,
    pub capture_state:
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
    // Create a new generator with the given sample frequency. Note that capture_state must be set
    // before `generate` is called.
    pub fn new(sample_rate: i32) -> Self {
        Generator {
            sample_rate,
            capture_state: None,
        }
    }

    // Generate a vector of samples up to `desired` length and return a new waveform that will continue where this
    // one left off. If fewer than 'desired' samples are generated, that indicates that this waveform has finished.
    // Waveforms should be initialized with `initialize_state` before the first call to `generate`. The returned
    // waveform supports being "re-initialized" -- it will generate the same samples each time it's initialized.
    // TODO take a &mut waveform instead
    pub fn generate<M: Debug + fmt::Display>(
        &self,
        waveform: &mut Waveform<M>,
        desired: usize,
    ) -> Vec<f32> {
        use State::*;
        use waveform::Waveform::*;
        if desired == 0 {
            return vec![];
        }
        match waveform {
            Const(value) => {
                vec![*value; desired]
            }
            Time(state @ Initial) => {
                *state = Position(0);
                self.generate(waveform, desired)
            }
            Time(Position(position)) => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (*position + i) as f32 / self.sample_rate as f32;
                }
                *position = *position + desired;
                out
            }
            Time(_) => unreachable!("Time waveform has non-Position state"),
            Noise => {
                let mut out = vec![0.0; desired];
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                out
            }
            Fixed(_, state @ Initial) => {
                *state = Position(0);
                self.generate(waveform, desired)
            }
            Fixed(samples, Position(position)) => {
                let mut out = vec![];
                if *position < samples.len() {
                    out.extend(&samples[*position..(*position + desired).min(samples.len())]);
                }
                *position = *position + out.len();
                out
            }
            Fixed(_, _) => unreachable!("Fixed waveform has non-Initial, non-Position state"),
            Fin { .. } => {
                // Set the inner waveform aside for a moment while we calculate the length.
                let Fin {
                    waveform: inner, ..
                } = waveform
                else {
                    unreachable!()
                };
                let tmp = mem::replace(inner, Box::new(Const(0.0)));
                // Note that this call to length also advances the position of the `length` waveform.
                let len = self.length(waveform, desired);
                // Restore the inner waveform.
                let Fin {
                    waveform: inner, ..
                } = waveform
                else {
                    unreachable!()
                };
                *inner = tmp;
                // Note that this might generate less than `len` since the call to `length` above
                // didn't account for the length of `inner`.
                self.generate(inner, len)
            }
            Append(a, b) => {
                let mut a_out = self.generate(a, desired);
                if a_out.len() == desired {
                    return a_out;
                }
                // a_out.len() < desired
                let mut b_out = self.generate(b, desired - a_out.len());
                a_out.append(&mut b_out);
                a_out
            }
            Sine {
                state: state @ Initial,
                ..
            } => {
                *state = Phase { accumulator: 0.0 };
                self.generate(waveform, desired)
            }
            Sine {
                frequency,
                phase,
                state: Phase { accumulator },
            } => {
                // Instantaneous frequency.
                let f_out = self.generate(frequency, desired);
                // Instantaneous phase offset.
                let ph_out = self.generate(phase, f_out.len());
                let mut out = vec![0.0; ph_out.len()];
                for (i, &phase_offset) in ph_out.iter().enumerate() {
                    out[i] = (*accumulator + phase_offset as f64).sin() as f32;

                    let f = f_out[i] as f64;
                    let phase_inc = f / self.sample_rate as f64;

                    // Move the accumulator according to the frequency and change in phase offset.
                    *accumulator = (*accumulator + phase_inc).rem_euclid(f64::consts::TAU);
                }
                out
            }
            Sine { .. } => unreachable!("Sine waveform has non-Initial, non-Phase state"),
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: state @ Initial,
            } => {
                // Generate the input samples so that they can be used in the feed-forward part of the filter.
                let ff_count = feed_forward.len();
                assert!(ff_count >= 1);
                let input = self.generate(inner, ff_count - 1);
                // Fill the previous output samples with zeros to match the number of feedback coefficients.
                let fb_count = feedback.len();
                let output = vec![0.0; fb_count];

                /*
                println!(
                    "Started generate for Filter with {} feed-forward and {} feedback coefficients",
                    ff_count, fb_count
                );
                */
                *state = Samples { input, output };
                self.generate(waveform, desired)
            }
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: Samples { input, output },
            } => {
                // We want the length of the output to match that of the input `inner`. Doing so means that
                // the last output point will only depend on one input point (and the  second to last output
                // will only depend on two input points, all the way back to the `ff_count-1`^th to last
                // point.) This means that if the inner waveform is not able to generate `desired` points,
                // we can zero-extend `input` up to `ff_count-1` points. We keep track of this by saving
                // fewer than ff_count points for the next call to `generate()`. This is the same thing that
                // happens when the first call to `generate()` with Initial state doesn't generate
                // `ff_count - 1` points.

                // Set up the input and output. Each will have extra samples at the beginning from a previous
                // call to `generate`. As noted above, `input` will be shorter than `ff_count - 1` in cases
                // where we've reached the end of the inner waveform. In those cases, we extend `input` with
                // up to `ff_count - 1` zeros.
                let ff_count = feed_forward.len();
                let (mut inner_out, input_padding) = if ff_count == input.len() + 1 {
                    let inner_out = self.generate(inner, desired);
                    let input_padding = (desired - inner_out.len()).min(ff_count - 1);
                    (inner_out, input_padding)
                } else {
                    // We've already exhausted `inner` on a previous call to generate (or in the Initial
                    // arm above). We will be able to generate `input.len()` more samples. To do that, we
                    // need to pad by `ff_count - 1`.
                    (vec![], ff_count - 1)
                };
                input.append(&mut inner_out);
                input.resize(input.len() + input_padding, 0.0);

                let fb_count = feedback.len();
                assert_eq!(fb_count, output.len());
                // Set the output length based on the size of the inner waveform.
                let mut out = output.clone();
                out.resize(input.len() - (ff_count - 1) + fb_count, 0.0);

                let mut recompute_coefficients = true;
                let mut ff_outs = vec![];
                let mut fb_outs = vec![];

                // Run the filter!!
                for i in fb_count..out.len() {
                    if recompute_coefficients {
                        recompute_coefficients = false;
                        ff_outs = Vec::with_capacity(ff_count);
                        fb_outs = Vec::with_capacity(fb_count);

                        // Gather the coefficients for this element of the output, zero-extending if necessary.
                        for w in feed_forward.iter_mut() {
                            if let Const(_) = w {
                            } else {
                                recompute_coefficients = true;
                            }
                            let mut ff_out = self.generate(w, 1);
                            ff_outs.push(ff_out.pop().unwrap_or(0.0));
                        }
                        for w in feedback.iter_mut() {
                            if let Const(_) = w {
                            } else {
                                recompute_coefficients = true;
                            }
                            let mut fb_out = self.generate(w, 1);
                            fb_outs.push(fb_out.pop().unwrap_or(0.0));
                        }
                        /*
                        println!(
                            "In generate() for Filter, generated coefficients:\n  b = {:?}\n  a = {:?}",
                            ff_outs, fb_outs
                        );
                        */
                    }

                    for (j, &ff) in ff_outs.iter().enumerate() {
                        out[i] += ff * input[i - fb_count + (ff_count - 1) - j];
                    }
                    for (j, &fb) in fb_outs.iter().enumerate() {
                        out[i] -= fb * out[i - j - 1];
                    }
                }

                // Save the last few samples of both the input and the output... but don't save any zeros in `input`
                // in the input that were padding on to the end.
                input.drain(0..input.len() - (ff_count - 1));
                input.truncate(input.len() - input_padding);
                *output = out[out.len() - fb_count..].to_vec();

                // Remove the fb_count samples from the beginning of `out` (these were included to compute
                // the feedback part of the filter).
                out.drain(0..fb_count);
                out
            }
            Filter { .. } => unreachable!("Filter waveform has non-Initial, non-Samples state"),
            BinaryPointOp(op, a, b) => {
                use waveform::Operator;
                let op_fn = match op {
                    Operator::Add | Operator::Merge => std::ops::Add::add,
                    Operator::Subtract => std::ops::Sub::sub,
                    Operator::Multiply => std::ops::Mul::mul,
                    Operator::Divide => |a: f32, b: f32| {
                        if b == 0.0 { 0.0 } else { a / b }
                    },
                };
                self.generate_binary_op(op_fn, a, b, *op == Operator::Merge, desired)
            }
            Reset {
                state: state @ Initial,
                ..
            } => {
                // Start assuming that the trigger was previously negative.
                *state = Sign { signum: -1.0 };
                self.generate(waveform, desired)
            }
            Reset {
                trigger,
                waveform: inner,
                state: Sign { signum },
            } => {
                let mut generated = 0;
                let mut out = Vec::new();

                let t_out = self.generate(trigger, desired);

                while generated < t_out.len() {
                    // Set to true if a restart will be triggered before desired is reached
                    let mut reset_inner_position = false;
                    let mut inner_desired = t_out.len() - generated;

                    for (i, &x) in t_out[generated..].iter().enumerate() {
                        if *signum < 0.0 && x >= 0.0 {
                            inner_desired = i;
                            reset_inner_position = true;
                            *signum = x.signum();
                            break;
                        } else if *signum >= 0.0 && x < 0.0 {
                            *signum = x.signum();
                        }
                    }

                    let mut tmp = self.generate(inner, inner_desired);
                    if tmp.len() < inner_desired {
                        tmp.resize(inner_desired, 0.0);
                    }
                    out.extend(tmp);
                    if reset_inner_position {
                        waveform::set_state(inner, State::Initial);
                    }
                    generated += inner_desired;
                }
                out
            }
            Reset { .. } => unreachable!("Reset waveform has non-Initial, non-Sign state"),
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let mut out = self.generate(trigger, desired);
                let mut positive_out = self.generate(positive_waveform, desired);
                positive_out.resize(out.len(), 0.0);
                let mut negative_out = self.generate(negative_waveform, desired);
                negative_out.resize(out.len(), 0.0);
                for (i, x) in out.iter_mut().enumerate() {
                    if *x >= 0.0 {
                        *x = positive_out[i];
                    } else {
                        *x = negative_out[i];
                    }
                }
                out
            }
            Marked {
                waveform: inner, ..
            } => self.generate(inner, desired),
            Captured {
                file_stem,
                waveform: inner,
            } => {
                // TODO think through this again
                //  - capture_state was set incorrectly when advancing position (i.e., when a waveform missed its start time)
                //  - we used to not generate the inner waveform when that was set... or was it unset?
                let out = self.generate(inner, desired);
                if self.capture_state.is_none() {
                    // This occurs, for example, when precomputing parts of a waveform.
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
                out
            }
        }
    }

    // Generate a binary operation on two waveforms, up to 'desired' samples. The `op_fn` function
    // is applied to each pair of samples. If `extend_to_longer` is true, the shorter waveform will
    // be extended with zeros to match the length of the longer one.
    fn generate_binary_op<M: Debug + fmt::Display>(
        &self,
        op_fn: fn(f32, f32) -> f32,
        a: &mut Waveform<M>,
        b: &mut Waveform<M>,
        extend_to_longer: bool,
        mut desired: usize,
    ) -> Vec<f32> {
        // Note that this approach may generate lots of samples from `a` that we don't need (in the
        // case that `b` is much shorter.)
        let mut a_out = self.generate(a, desired);
        match b {
            // In this branch (which is always taken if we've removed Seq's), check to see if we can
            // avoid generating the right-hand side and instead just apply the op directly.
            // TODO could also check f against the identity of op and skip the loop here
            waveform::Waveform::Const(f) => {
                for x in a_out.iter_mut() {
                    *x = op_fn(*x, *f);
                }
                if desired > a_out.len() && extend_to_longer {
                    a_out.resize(desired, op_fn(0.0, *f));
                }
                // In theory, we need to advance `b`, but that's a no-op given that it's a Const
                a_out
            }
            _ => {
                if !extend_to_longer {
                    // This will mean that `b` is no longer than `a`.
                    desired = a_out.len();
                }
                let mut b_out = self.generate(b, desired);
                // Merge the overlapping portion
                for (i, x) in a_out.iter_mut().enumerate() {
                    if i >= b_out.len() {
                        break;
                    }
                    *x = op_fn(*x, b_out[i]);
                }
                // At this point we've updated all of the elements up to the minimum of the two
                // lengths.
                // If the right side is shorter, then truncate.
                if !extend_to_longer && b_out.len() < a_out.len() {
                    a_out.truncate(b_out.len());
                }
                // If the left side is shorter than the right, then append.
                if extend_to_longer && b_out.len() > a_out.len() {
                    b_out.drain(0..a_out.len());
                    a_out.append(&mut b_out);
                }
                a_out
            }
        }
    }

    // Returns the number of samples that `waveform` will generate or `max`, whichever is
    // smaller, while modifying `waveform` so that it has been advanced to that point.
    // Note that for waveforms that maintain state (other than position), the state is passed
    // through unchanged. This discontinuity may result in pops or clicks for audible waveforms.
    pub fn length<M: Debug + fmt::Display>(&self, waveform: &mut Waveform<M>, max: usize) -> usize {
        use State::*;
        use waveform::Operator;
        use waveform::Waveform::*;
        match waveform {
            Const { .. } => max,
            Time(state @ Initial) => {
                *state = Position(0);
                self.length(waveform, max)
            }
            Time(Position(position)) => {
                *position = *position + max;
                max
            }
            Time(_) => unreachable!("Time waveform with non-Initial, non-Position state"),
            Noise => max,
            Fixed(_, state @ Initial) => {
                *state = Position(0);
                self.length(waveform, max)
            }
            Fixed(samples, Position(position)) => {
                if *position >= samples.len() {
                    return 0;
                }
                let len = max.min(samples.len() - *position);
                *position = *position + len;
                return len;
            }
            Fixed(_, _) => unreachable!("Fixed waveform with non-Initial, non-Position state"),
            Fin {
                length,
                waveform: inner,
            } => {
                // This is a little subtle, since Fin is not supposed to make waveforms longer.
                // In particular, for optimizations that move Fin outside of a DotProduct, we need
                // to check the length of the inner waveform to see if it's shorter than `length`
                // would indicate.
                match self.greater_or_equals_at(length, 0.0, max) {
                    MaybeOption::Some(len) => {
                        // Check to see if `inner` is shorter and use that result.
                        let inner_len = self.length(inner, len);
                        // Finally, advance `length` too
                        let _ = self.length(length, inner_len);
                        inner_len
                    }
                    MaybeOption::None => {
                        // Check to see if `inner` is shorter and use that result.
                        let inner_len = self.length(inner, max);
                        // Advance `length` by `inner_len` samples
                        let _ = self.length(length, inner_len);
                        inner_len
                    }
                    MaybeOption::Maybe => {
                        println!(
                            "Warning: unable to determine root of Fin length cheaply, generating samples for: {:?}",
                            length
                        );
                        for i in 0..max {
                            let length_out = self.generate(length, 1);
                            if length_out.len() == 0 {
                                return i;
                            }
                            // Advance inner by one sample and check to see if it's finished.
                            let inner_len = self.length(inner, 1);
                            if inner_len == 0 {
                                return i;
                            }
                            if length_out[0] >= 0.0 {
                                return i + 1;
                            }
                        }
                        // Note that `new_length` and `new_inner` are already advanced by `max` samples
                        max
                    }
                }
            }
            Filter {
                waveform,
                feed_forward,
                feedback,
                state: state @ Initial,
            } => {
                let ff_count = feed_forward.len();
                let fb_count = feedback.len();
                assert!(ff_count >= 1);
                *state = Samples {
                    input: vec![0.0; ff_count - 1],
                    output: vec![0.0; fb_count],
                };
                self.length(waveform, max)
            }
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: Samples { .. },
            } => {
                // TODO I think this logic could be made to better parallel that of generate() (by actually advancing in the Initial arm and "padding" here.)
                let inner_len = self.length(inner, max);
                // Only the length of inner matters, but we need to advance all of the coefficient waveforms
                for w in feed_forward.iter_mut() {
                    let _ = self.length(w, inner_len);
                }
                for w in feedback.iter_mut() {
                    let _ = self.length(w, inner_len);
                }
                inner_len
            }
            Filter { .. } => {
                unreachable!("Filter with non-Initial, non-Samples state")
            }
            Append(a, b) => {
                let a_len = self.length(a, max);
                let b_len = self.length(b, max - a_len);
                a_len + b_len
            }
            Sine {
                frequency, phase, ..
            } => {
                let f_len = self.length(frequency, max);
                let ph_len = self.length(phase, max);
                f_len.min(ph_len)
            }
            BinaryPointOp(op, a, b) => {
                let a_len = self.length(a, max);
                let b_len = self.length(b, max);
                match op {
                    Operator::Add | Operator::Subtract | Operator::Multiply | Operator::Divide => {
                        a_len.min(b_len)
                    }
                    Operator::Merge => a_len.max(b_len),
                }
            }
            Reset { trigger, .. } => {
                let len = self.length(trigger, max);
                // We don't change the state of waveform here as its position
                // isn't meaningful in a global sense.
                len
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let len = self.length(trigger, max);
                // Advance the position of the positive and negative waveforms.
                let _ = self.length(positive_waveform, len);
                let _ = self.length(negative_waveform, len);
                len
            }
            Marked { waveform, .. } => {
                let len = self.length(waveform, max);
                len
            }
            Captured { waveform, .. } => {
                let len = self.length(waveform, max);
                len
            }
        }
    }

    // If `waveform` will be greater than or equal to `value` at some point between its current position and
    // `max`, return Some of the number of samples that would be generated before then, None if `waveform`
    // will not be greater than or equal in that range, or Maybe if that can't be determined cheaply.
    fn greater_or_equals_at<M: fmt::Display>(
        &self,
        waveform: &Waveform<M>,
        value: f32,
        max: usize,
    ) -> MaybeOption<usize> {
        use State::{Initial, Position};
        use waveform::Operator;
        use waveform::Waveform::{Append, BinaryPointOp, Const, Time};
        match waveform {
            Const(v) if *v >= value => MaybeOption::Some(0),
            Const(_) => MaybeOption::None,
            Time(Initial) => self.greater_or_equals_at::<M>(&Time(Position(0)), value, max),
            Time(Position(position)) => {
                let current_value = *position as f32 / self.sample_rate as f32;
                if current_value >= value {
                    MaybeOption::Some(0)
                } else {
                    // current_value < value and current_value >= 0 so value > 0 (so usize is ok)
                    let target_position = (value * self.sample_rate as f32).ceil() as usize;
                    // Also, target_position must be > position
                    MaybeOption::Some(max.min((target_position - position) as usize))
                }
            }
            Append(a, _) => {
                match self.greater_or_equals_at(a, value, max) {
                    MaybeOption::Some(size) => MaybeOption::Some(size),
                    MaybeOption::None => {
                        /* XXX do we need this? for all cases? (maybe just for a = Fixed?)
                        let a_len = self.length(*a.clone(), max);
                        if a_len == max {
                            // We didn't reach the end of `a`, so `b` isn't relevant yet.
                            MaybeOption::None
                        } else {
                            match self.greater_or_equals_at(&b, value, max - a_len) {
                                MaybeOption::Some(size) => MaybeOption::Some(size + a_len),
                                m => m,
                            }
                        }
                        */
                        println!("Warning: in greater_or_equals_at for Append... returning Maybe");
                        MaybeOption::Maybe
                    }
                    m => m, // Maybe gets passed through
                }
            }
            BinaryPointOp(op @ (Operator::Add | Operator::Subtract), a, b) => {
                use waveform::Operator::{Add, Subtract};
                match (op, a.as_ref(), b.as_ref()) {
                    // TODO need to consider constant functions as const
                    (Add, Const(va), Const(vb)) if va + vb >= value => MaybeOption::Some(0),
                    (Add, Const(_), Const(_)) => MaybeOption::None,
                    (Add, Const(va), _) => self.greater_or_equals_at(b, value - va, max),
                    (Add, _, Const(vb)) => self.greater_or_equals_at(a, value - vb, max),

                    (Subtract, Const(va), Const(vb)) if va - vb >= value => MaybeOption::Some(0),
                    (Subtract, Const(_), Const(_)) => MaybeOption::None,
                    (Subtract, _, Const(vb)) => self.greater_or_equals_at(a, value + vb, max),

                    _ => MaybeOption::Maybe,
                }
            }
            // TODO think about Marked here
            _ => {
                println!("Unhandled case in greater_or_equals_at: {}", waveform);
                MaybeOption::Maybe
            }
        }
    }

    // Replaces parts of `waveform` that can be precomputed with their equivalent Fixed versions. Notably,
    // infinite waveforms and waveforms that depend on or have dynamic behavior (Marked, Captured)
    // cannot be replaced.
    pub fn precompute<M>(&self, waveform: Waveform<M>) -> Waveform<M>
    where
        M: Clone + Debug + fmt::Display,
    {
        #[derive(Clone, Copy)]
        enum Reason {
            // The waveform is not pre-computable because it is infinite in length.
            Infinite,
            // The waveform is not pre-computable because it depends on some run-time state or has a run-time effect.
            Dynamic,
        }
        enum Result<M> {
            // Pre-computable
            PC(Waveform<M>),
            // Not-pre-computable
            NPC(Reason, Waveform<M>),
        }

        impl<M> Into<Waveform<M>> for Result<M> {
            fn into(self) -> Waveform<M> {
                match self {
                    Result::PC(w) => w,
                    Result::NPC(_, w) => w,
                }
            }
        }

        // generate_fixed generates `waveform` up to some large number of samples. It should only be used on waveforms
        // that are pre-computable.
        fn generate_fixed<M>(g: &Generator, mut waveform: Waveform<M>) -> Waveform<M>
        where
            M: Debug + fmt::Display,
        {
            // Choose a `desired` which is long enough to generate any reasonable waveform, but give some room
            // for cases like `Filter` that may need to make it longer.

            println!("Precomputing output for {}", &waveform);
            let max_desired = (g.sample_rate * 10) as usize;
            let out = g.generate(&mut waveform, max_desired);
            if out.len() == max_desired {
                println!("Warning: precompute generated max samples (maybe not finite?)");
            }

            println!("  ...generated {} samples", out.len());

            waveform::Waveform::Fixed(out, State::Initial)
        }

        fn precompute_internal<M>(g: &Generator, waveform: Waveform<M>) -> Result<M>
        where
            M: Debug + fmt::Display,
        {
            use Reason::*;
            use Result::*;
            use waveform::Operator;
            use waveform::Waveform::*;

            // do_one_dynamic takes a waveform and determines if it can be pre-computed. If so, it applies `wf` and
            // then wraps the result in NPC(Dynamic).
            fn do_one_dynamic<M, F: FnOnce(Waveform<M>) -> Waveform<M>>(
                g: &Generator,
                a: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + fmt::Display,
            {
                match precompute_internal(g, a) {
                    PC(a) => NPC(Dynamic, wf(generate_fixed(g, a))),
                    NPC(_, a) => NPC(Dynamic, wf(a)),
                }
            }

            // do_two attempts to pre-compute two waveforms, then applies `wf` to the results. If both waveforms are
            // pre-computable, then the result is wrapped in PC. If at least one is not pre-computable, then the result
            // is wrapped in NPC, with the reason determined by the reason(s) for the two waveforms.
            fn do_two<M, F: FnOnce(Waveform<M>, Waveform<M>) -> Waveform<M>>(
                g: &Generator,
                a: Waveform<M>,
                b: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + fmt::Display,
            {
                match (precompute_internal(g, a), precompute_internal(g, b)) {
                    (PC(a), PC(b)) => PC(wf(a, b)),
                    (PC(a), NPC(why, b)) => NPC(why, wf(generate_fixed(g, a), b)),
                    (NPC(why, a), PC(b)) => NPC(why, wf(a, generate_fixed(g, b))),
                    (NPC(Infinite, a), NPC(Infinite, b)) => NPC(Infinite, wf(a, b)),
                    // At least one is Dynamic and neither is pre-computable
                    (a, b) => NPC(Dynamic, wf(a.into(), b.into())),
                }
            }

            fn resolve_reason(why1: Reason, why2: Reason) -> Reason {
                // If either one is Dynamic then the result should be too
                match (why1, why2) {
                    (Infinite, Infinite) => Infinite,
                    _ => Dynamic,
                }
            }

            // Like do_two, do_three, attempts to pre-compute the three waveforms, then applies `wf` to the results.
            // The result is wrapped in PC if all three are pre-computable, and NPC otherwise, with the reason
            // determined by the reason(s) for the three waveforms.
            fn do_three<M, F: FnOnce(Waveform<M>, Waveform<M>, Waveform<M>) -> Waveform<M>>(
                g: &Generator,
                a: Waveform<M>,
                b: Waveform<M>,
                c: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + fmt::Display,
            {
                match (
                    precompute_internal(g, a),
                    precompute_internal(g, b),
                    precompute_internal(g, c),
                ) {
                    // All pre-computable
                    (PC(a), PC(b), PC(c)) => PC(wf(a, b, c)),

                    // One is not pre-computable, two are pre-computable, so pre-compute those two now
                    (PC(a), PC(b), NPC(why, c)) => {
                        NPC(why, wf(generate_fixed(g, a), generate_fixed(g, b), c))
                    }
                    (PC(a), NPC(why, b), PC(c)) => {
                        NPC(why, wf(generate_fixed(g, a), b, generate_fixed(g, c)))
                    }
                    (NPC(why, a), PC(b), PC(c)) => {
                        NPC(why, wf(a, generate_fixed(g, b), generate_fixed(g, c)))
                    }

                    // Two are not pre-computable, one is pre-computable, so pre-compute that one now
                    (NPC(why1, a), NPC(why2, b), PC(c)) => {
                        NPC(resolve_reason(why1, why2), wf(a, b, generate_fixed(g, c)))
                    }
                    (NPC(why1, a), PC(b), NPC(why2, c)) => {
                        NPC(resolve_reason(why1, why2), wf(a, generate_fixed(g, b), c))
                    }
                    (PC(a), NPC(why1, b), NPC(why2, c)) => {
                        NPC(resolve_reason(why1, why2), wf(generate_fixed(g, a), b, c))
                    }

                    // All three are not pre-computable
                    (NPC(why1, a), NPC(why2, b), NPC(why3, c)) => NPC(
                        resolve_reason(resolve_reason(why1, why2), why3),
                        wf(a, b, c),
                    ),
                }
            }

            match waveform {
                // Const, Time, and Noise are all infinite.
                Const(_) | Time(_) | Noise => NPC(Infinite, waveform),
                // Fixed is the quintessential pre-computable waveform.
                Fixed(_, _) => PC(waveform),
                Fin { length, waveform } => match (
                    // XXX we could check to see that `length` crosses zero at some point for the cases where we
                    // call generate_fixed
                    precompute_internal(g, *length),
                    precompute_internal(g, *waveform),
                ) {
                    (length, NPC(Dynamic, waveform)) => {
                        println!(
                            "Cannot precompute Fin because inner waveform is dynamic: {}",
                            &waveform
                        );
                        NPC(
                            Dynamic,
                            Fin {
                                length: Box::new(length.into()),
                                waveform: Box::new(waveform),
                            },
                        )
                    }
                    (NPC(Dynamic, length), waveform) => {
                        println!(
                            "Cannot precompute Fin because length waveform is dynamic: {}",
                            &length
                        );
                        NPC(
                            Dynamic,
                            Fin {
                                length: Box::new(length),
                                waveform: Box::new(waveform.into()),
                            },
                        )
                    }
                    // Neither is dynamic, so we can pre-compute
                    // TODO... maybe we should check that `length` is non-zero at some point?
                    (length, waveform) => PC(Fin {
                        length: Box::new(length.into()),
                        waveform: Box::new(waveform.into()),
                    }),
                },
                Append(a, b) => do_two(g, *a, *b, |a, b| Append(Box::new(a), Box::new(b))),
                Sine {
                    frequency,
                    phase,
                    state,
                } => do_two(g, *frequency, *phase, |frequency, phase| Sine {
                    frequency: Box::new(frequency),
                    phase: Box::new(phase),
                    state,
                }),
                BinaryPointOp(op, a, b) => {
                    match (op, precompute_internal(g, *a), precompute_internal(g, *b)) {
                        (op, PC(a), PC(b)) => PC(BinaryPointOp(op, Box::new(a), Box::new(b))),
                        (op @ (Operator::Multiply | Operator::Divide), NPC(Infinite, a), PC(b))
                        | (op @ (Operator::Multiply | Operator::Divide), PC(a), NPC(Infinite, b)) => {
                            PC(BinaryPointOp(op, Box::new(a), Box::new(b)))
                        }
                        (op, PC(a), NPC(why, b)) => NPC(
                            why,
                            BinaryPointOp(op, Box::new(generate_fixed(g, a)), Box::new(b)),
                        ),
                        (op, NPC(why, a), PC(b)) => NPC(
                            why,
                            BinaryPointOp(op, Box::new(a), Box::new(generate_fixed(g, b))),
                        ),
                        (op, NPC(Infinite, a), NPC(Infinite, b)) => {
                            NPC(Infinite, BinaryPointOp(op, Box::new(a), Box::new(b)))
                        }
                        // At least one is Dynamic and neither is pre-computable
                        (op, a, b) => NPC(
                            Dynamic,
                            BinaryPointOp(op, Box::new(a.into()), Box::new(b.into())),
                        ),
                    }
                }
                Filter {
                    waveform,
                    feed_forward,
                    feedback,
                    state,
                } => {
                    let inner = precompute_internal(g, *waveform);
                    let ff: Vec<_> = feed_forward
                        .into_iter()
                        .map(|w| precompute_internal(g, w))
                        .collect();
                    let fb: Vec<_> = feedback
                        .into_iter()
                        .map(|w| precompute_internal(g, w))
                        .collect();

                    let mut reason: Option<Reason> = None;
                    if let NPC(why, _) = &inner {
                        reason = Some(*why);
                    }
                    for r in &ff {
                        if let NPC(why, _) = r {
                            reason = Some(match reason {
                                Some(prev) => resolve_reason(prev, *why),
                                None => *why,
                            });
                        }
                    }
                    for r in &fb {
                        if let NPC(why, _) = r {
                            reason = Some(match reason {
                                Some(prev) => resolve_reason(prev, *why),
                                None => *why,
                            });
                        }
                    }

                    // If `reason` is Some then we will be returning NPC for the Filter, so generate
                    // samples for any pre-computable sub-waveforms.
                    let extract = |r: Result<M>| match (r, &reason) {
                        (PC(w), Some(_)) => generate_fixed(g, w),
                        (PC(w), None) => w,
                        (NPC(_, w), _) => w,
                    };
                    let inner_wf = extract(inner);
                    let ff_wfs: Vec<_> = ff.into_iter().map(|r| extract(r)).collect();
                    let fb_wfs: Vec<_> = fb.into_iter().map(|r| extract(r)).collect();

                    let filter_wf = Filter {
                        waveform: Box::new(inner_wf),
                        feed_forward: ff_wfs,
                        feedback: fb_wfs,
                        state,
                    };
                    match reason {
                        Some(why) => NPC(why, filter_wf),
                        None => PC(filter_wf),
                    }
                }
                Reset {
                    trigger,
                    waveform,
                    state,
                } => do_two(g, *trigger, *waveform, |trigger, waveform| Reset {
                    trigger: Box::new(trigger),
                    waveform: Box::new(waveform),
                    state,
                }),
                Alt {
                    trigger,
                    positive_waveform,
                    negative_waveform,
                } => do_three(
                    g,
                    *trigger,
                    *positive_waveform,
                    *negative_waveform,
                    |trigger, positive_waveform, negative_waveform| Alt {
                        trigger: Box::new(trigger),
                        positive_waveform: Box::new(positive_waveform),
                        negative_waveform: Box::new(negative_waveform),
                    },
                ),
                // Marked and Captured may pre-compute their inner waveforms, but they themselves are still dynamic.
                Marked { waveform, id } => do_one_dynamic(g, *waveform, |waveform| Marked {
                    waveform: Box::new(waveform),
                    id,
                }),
                Captured {
                    waveform,
                    file_stem,
                } => do_one_dynamic(g, *waveform, |waveform| Captured {
                    waveform: Box::new(waveform),
                    file_stem,
                }),
            }
        }

        let result = match precompute_internal(self, waveform.clone()) {
            Result::PC(w) => generate_fixed(self, w),
            Result::NPC(_, w) => w,
        };
        /*
        println!(
            "precompute waveform:\n{:#?}\ninto:\n{:#?}",
            &waveform, &result
        );
        */
        return result;
    }
}

#[cfg(test)]
mod tests {
    use std::f32;

    use super::*;
    use crate::optimizer;
    use waveform::Operator;
    use waveform::Waveform::{Append, BinaryPointOp, Const, Filter, Fin, Fixed, Reset, Sine, Time};

    type Waveform = waveform::Waveform<u32>;

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

    // Verifies that `waveform` would generate `expected` samples using a call to `length`.
    fn check_length(
        g: &Generator,
        waveform: &Waveform,
        position: usize,
        expected: usize,
        max: usize,
    ) {
        let mut waveform = initialize_state(waveform.clone());
        // Generate up to `position` and discard those samples.
        let _ = g.generate(&mut waveform, position);
        assert_eq!(
            g.length(&mut waveform.clone(), max),
            expected,
            "Expected length {} (with max = {}) for waveform: {:?}",
            expected,
            max,
            waveform
        );
    }

    fn run_tests(waveform: &Waveform, desired: &Vec<f32>) {
        let g = new_test_generator(1);
        // Check that `waveform` would generate at least as many samples as `desired`.
        check_length(&g, &waveform, 0, desired.len(), desired.len());

        for size in [1, 2, 4, 8] {
            let mut w = initialize_state(waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = g.generate(&mut w, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let tmp = g.generate(&mut w, out.len() % size);
                let n = (out.len() - 1) / size;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, *desired,
                "Failed output for size {} on waveform\n{:#?}",
                size, waveform
            );
        }

        for size in [1, 2, 4, 8] {
            let optimized_waveform = optimizer::optimize(waveform.clone());
            check_length(&g, &optimized_waveform, 0, desired.len(), desired.len());
            let mut w = initialize_state(optimized_waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = g.generate(&mut w, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let tmp = g.generate(&mut w, out.len() % size);
                let n = (out.len() - 1) / size;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, *desired,
                "Failed output for size {} on waveform\n{:#?}\noptimized to\n{:#?}",
                size, waveform, optimized_waveform
            );
        }

        for size in [1, 2, 4, 8] {
            let g = new_test_generator(1);
            let optimized_waveform = optimizer::optimize(waveform.clone());
            let precomputed_waveform = g.precompute(initialize_state(optimized_waveform));
            let mut w = precomputed_waveform.clone();
            // XXX check_length(&g, &w, 0, desired.len(), desired.len());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let tmp = g.generate(&mut w, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let tmp = g.generate(&mut w, out.len() % size);
                let n = (out.len() - 1) / size;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, *desired,
                "Failed output for size {} on waveform\n{:#?}\nprecomputed to\n{:#?}",
                size, waveform, precomputed_waveform
            );
        }
    }

    #[test]
    fn test_time() {
        let w = Time(());
        run_tests(&w, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_fixed() {
        let w = Fixed(vec![1.0, 2.0, 3.0, 4.0, 5.0], ());
        run_tests(&w, &vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let g = new_test_generator(1);
        let mut w = initialize_state(w);
        // Advance past the end of the waveform
        let _ = g.generate(&mut w, 6);
        let result = g.generate(&mut w, 8);
        assert_eq!(result, vec![] as Vec<f32>);
    }

    // frequency in Hz, phase in radians
    fn sin_waveform(frequency: f32, phase: f32) -> Box<Waveform> {
        return Box::new(Sine {
            frequency: Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(f32::consts::TAU)),
                Box::new(Const(frequency)),
            )),
            phase: Box::new(Const(phase)),
            state: (),
        });
    }

    fn run_sin_test<M>(g: &Generator, waveform: &mut super::Waveform<M>, expected: Vec<f32>)
    where
        M: Debug + fmt::Display,
    {
        let result = g.generate(waveform, expected.len());
        for (i, &x) in result.iter().enumerate() {
            assert!(
                (x - expected[i]).abs() < 1e-5,
                "result = {}, expected = {}, result - expected = {} at index {}",
                x,
                expected[i],
                x - expected[i],
                i
            );
        }
    }

    #[test]
    fn test_sin() {
        let sample_frequency = 44100.0;
        let g = new_test_generator(sample_frequency as i32);

        // As simple as possible
        let mut w = initialize_state(*sin_waveform(1.0, 0.0));
        let expected = (0..100)
            .map(|x: i32| (f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&g, &mut w, expected);

        // Non-constant frequency: f = time + 10 Hz
        let mut w = initialize_state(Waveform::Sine {
            frequency: Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Time(())),
                    Box::new(Const(10.0)),
                )),
                Box::new(Const(f32::consts::TAU)),
            )),
            phase: Box::new(Const(0.0)),
            state: (),
        });
        let f_is_t_plus_ten = |x: i32| {
            let t = x as f64 / sample_frequency;
            let phase = f64::consts::TAU * (0.5 * t * t + 10.0 * t);
            phase.sin() as f32
        };
        let expected = (0..100).map(f_is_t_plus_ten).collect();
        run_sin_test(&g, &mut w, expected);

        // Non-zero phase offset
        let mut w = initialize_state(*sin_waveform(0.25, f32::consts::PI));
        let expected = (0..100)
            .map(|x| {
                (f64::consts::TAU * 0.25 * x as f64 / sample_frequency + f64::consts::PI).sin()
                    as f32
            })
            .collect();
        run_sin_test(&g, &mut w, expected);
    }

    #[test]
    fn test_reset() {
        let w = Reset {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w, &vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w = Reset {
            trigger: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(6.0)),
                )),
                waveform: sin_waveform(0.25, 0.0),
            }),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w, &vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

        let w = Reset {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(3.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            state: (),
        };
        run_tests(&w, &vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);

        let w = Reset {
            trigger: sin_waveform(0.25, f32::consts::PI),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w, &vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

        // Test where a reset lines up with the buffer boundary and where there are multiple
        // resets in a buffer.
        let w = Reset {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(
            &w,
            &vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
            ],
        );
    }

    #[test]
    fn test_append() {
        let g = new_test_generator(1);
        let w = Append(
            Box::new(Fixed(vec![1.0; 3], ())),
            Box::new(Fixed(vec![2.0; 3], ())),
        );

        check_length(&g, &w, 0, 6, MAX_LENGTH);
        check_length(&g, &w, 2, 4, MAX_LENGTH);
        check_length(&g, &w, 4, 2, MAX_LENGTH);
        run_tests(&w, &vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        match g.precompute(initialize_state(w)) {
            Fixed(_, _) => (), // Already checked the result above
            w => panic!(
                "Expected the Append to be precomputed to a Fixed, but got {:?}",
                w
            ),
        }
    }

    #[test]
    fn test_sum() {
        // Add yields a result as long as the shorter of the two inputs.

        // Both infinite
        let w = BinaryPointOp(Operator::Add, Box::new(Const(1.0)), Box::new(Const(2.0)));
        run_tests(&w, &vec![3.0; 8]);

        // One finite, one infinite: truncates to the finite length
        let w = BinaryPointOp(
            Operator::Add,
            Box::new(Fixed(vec![1.0, 2.0, 3.0], ())),
            Box::new(Const(10.0)),
        );
        run_tests(&w, &vec![11.0, 12.0, 13.0]);

        // Both finite, different lengths: truncates to the shorter
        let w = BinaryPointOp(
            Operator::Add,
            Box::new(Fixed(vec![1.0, 2.0], ())),
            Box::new(Fixed(vec![10.0, 20.0, 30.0], ())),
        );
        run_tests(&w, &vec![11.0, 22.0]);

        // Fin + Const
        let w = Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time(())),
                Box::new(Const(4.0)),
            )),
            waveform: Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(1.0)),
                Box::new(Const(2.0)),
            )),
        };
        run_tests(&w, &vec![3.0, 3.0, 3.0, 3.0]);

        // Add with Fixed([], ()) yields empty (shorter is 0)
        let w = BinaryPointOp(
            Operator::Add,
            Box::new(Fixed(vec![], ())),
            Box::new(Const(5.0)),
        );
        run_tests(&w, &vec![]);
    }

    #[test]
    fn test_dot_product() {
        let g = new_test_generator(1);

        // Multiply yields a result as long as the shorter of the two inputs.

        // Both infinite
        let w = Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time(())),
                Box::new(Const(8.0)),
            )),
            waveform: Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(3.0)),
                Box::new(Const(2.0)),
            )),
        };
        run_tests(&w, &vec![6.0; 8]);

        // One finite, one infinite: truncates to finite
        let w = BinaryPointOp(
            Operator::Multiply,
            Box::new(Fixed(vec![3.0, 4.0, 5.0], ())),
            Box::new(Const(2.0)),
        );
        run_tests(&w, &vec![6.0, 8.0, 10.0]);

        // Both finite, different lengths: truncates to shorter
        let w = BinaryPointOp(
            Operator::Multiply,
            Box::new(Fixed(vec![3.0, 4.0], ())),
            Box::new(Fixed(vec![2.0, 5.0, 1.0], ())),
        );
        run_tests(&w, &vec![6.0, 20.0]);

        // Multiply with empty Fixed yields empty
        let w = BinaryPointOp(
            Operator::Multiply,
            Box::new(Fixed(vec![], ())),
            Box::new(Const(5.0)),
        );
        run_tests(&w, &vec![]);

        // Precompute: finite * Const should precompute to Fixed
        let w = BinaryPointOp(
            Operator::Multiply,
            Box::new(Fixed(vec![3.0, 4.0, 5.0], ())),
            Box::new(Const(2.0)),
        );
        run_tests(&w, &vec![6.0, 8.0, 10.0]);
        match g.precompute(initialize_state(w)) {
            Fixed(_, _) => (),
            w => panic!(
                "Expected the result to be precomputed to a Fixed, but got {:?}",
                w
            ),
        }
    }

    #[test]
    fn test_merge() {
        // Merge behaves like Add but yields a result as long as the longer of the two inputs.

        // Both infinite
        let w = BinaryPointOp(Operator::Merge, Box::new(Const(1.0)), Box::new(Const(2.0)));
        run_tests(&w, &vec![3.0; 8]);

        // Both finite, different lengths: extends to the longer
        let w = BinaryPointOp(
            Operator::Merge,
            Box::new(Fixed(vec![1.0, 2.0], ())),
            Box::new(Fixed(vec![10.0, 20.0, 30.0], ())),
        );
        run_tests(&w, &vec![11.0, 22.0, 30.0]);

        // One finite, one infinite: extends to infinite
        let w = BinaryPointOp(
            Operator::Merge,
            Box::new(Fixed(vec![1.0, 2.0], ())),
            Box::new(Const(10.0)),
        );
        run_tests(&w, &vec![11.0, 12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);

        // Both finite, same length
        let w = BinaryPointOp(
            Operator::Merge,
            Box::new(Fixed(vec![1.0, 2.0], ())),
            Box::new(Fixed(vec![10.0, 20.0], ())),
        );
        run_tests(&w, &vec![11.0, 22.0]);

        // Merge with empty Fixed: the other side survives
        let w5 = BinaryPointOp(
            Operator::Merge,
            Box::new(Fixed(vec![], ())),
            Box::new(Fixed(vec![10.0, 20.0], ())),
        );
        run_tests(&w5, &vec![10.0, 20.0]);
    }

    #[test]
    fn test_filter() {
        let g = new_test_generator(1);

        // FIRs
        let w = Filter {
            waveform: Box::new(Time(())),
            feed_forward: vec![Const(2.0), Const(2.0), Const(2.0)],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]);

        let w = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(5.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            feed_forward: vec![Const(2.0), Const(2.0), Const(2.0)],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![6.0, 12.0, 18.0, 14.0, 8.0]);

        let w = Filter {
            waveform: Box::new(Fixed(vec![1.0, 2.0, 3.0], ())),
            feed_forward: vec![Const(2.0), Const(2.0), Const(2.0), Const(2.0), Const(2.0)],
            feedback: vec![],
            state: (),
        };
        check_length(&g, &w, 0, 3, 5);

        let w = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(8.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            feed_forward: vec![Const(2.0), Const(2.0), Const(2.0), Const(2.0), Const(2.0)],
            feedback: vec![],
            state: (),
        };
        let g = new_test_generator(1);
        check_length(&g, &w, 0, 8, MAX_LENGTH);
        run_tests(&w, &vec![20.0, 30.0, 40.0, 50.0, 44.0, 36.0, 26.0, 14.0]);

        let w = Filter {
            waveform: Box::new(Reset {
                // Pick a trigger that's far from zero on at our sampled points
                trigger: sin_waveform(1.0 / 3.0, 3.0 * std::f32::consts::PI / 2.0),
                waveform: Box::new(Time(())),
                state: (),
            }),
            feed_forward: vec![Const(2.0), Const(2.0)],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![0.0, 2.0, 6.0, 4.0, 2.0, 6.0, 4.0, 2.0]);

        let w = Filter {
            waveform: Box::new(Const(1.0)),
            feed_forward: vec![Const(0.2), Const(0.2), Const(0.2), Const(0.2), Const(0.2)],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // IIRs
        let w = Filter {
            waveform: Box::new(Time(())),
            feed_forward: vec![Const(0.5)],
            feedback: vec![Const(-0.5)],
            state: (),
        };
        run_tests(
            &w,
            &vec![0.0, 0.5, 1.25, 2.125, 3.0625, 4.03125, 5.015625, 6.0078125],
        );

        // Cascade
        let w = Filter {
            waveform: Box::new(Filter {
                waveform: Box::new(Time(())),
                feed_forward: vec![Const(0.5)],
                feedback: vec![Const(-0.5)],
                state: (),
            }),
            feed_forward: vec![Const(0.4)],
            feedback: vec![Const(-0.6)],
            state: (),
        };
        run_tests(
            &w,
            &vec![
                0.0, 0.2, 0.62, 1.222, 1.9582, 2.7874203, 3.6787024, 4.610347,
            ],
        );

        // Input is Const(1.0), so out[n] = 1.0*1.0 + n*1.0 = n+1.
        let w = Filter {
            waveform: Box::new(Const(1.0)),
            feed_forward: vec![Const(1.0), Time(())],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let w = Filter {
            waveform: Box::new(Fixed(vec![1.0; 3], ())),
            feed_forward: vec![Const(1.0), Fixed(vec![2.0], ()), Fixed(vec![3.0; 2], ())],
            feedback: vec![],
            state: (),
        };
        run_tests(&w, &vec![6.0, 3.0, 0.0]);

        // TODO add test where length of inner waveform is less than ff_count - 1
    }

    #[test]
    fn test_greater_or_equals_at() {
        let w1: Waveform = BinaryPointOp(Operator::Add, Box::new(Time(())), Box::new(Const(-5.0)));
        let w2: Waveform = Fin {
            length: Box::new(w1.clone()),
            waveform: Box::new(Time(())),
        };
        let g = new_test_generator(1);
        let position = g.greater_or_equals_at(&initialize_state(w1.clone()), 0.0, 10);
        let out = g.generate(&mut initialize_state(w2), 10);
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
