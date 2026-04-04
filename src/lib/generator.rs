use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Display};
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
    // TODO consider a shared vec plus a slice for Fixed?
    // Shared(Rc<Vec<f32>>, &[f32]),
    // Previously generated samples as input and/or output to the waveform (for example, for Filter)
    Samples {
        input: VecDeque<f32>,
        output: VecDeque<f32>,
    },
    // The current accumulated phase (for example, for Sine).
    Phase {
        accumulator: f64,
    },
    // The sign of the last generated value along with the direction of generation (for example, for Reset).
    Sign {
        signum: f32,
    },
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
    // Total number of samples allocated as part of generation
    pub allocations: usize,
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
            allocations: 0,
        }
    }

    /// Generate a vector of samples into `out` while modifying `waveform` so that subsequent calls to
    /// generate will pick up where this one left off.
    ///
    /// Returns the number of samples generated. If that is less than the length of `out`, that indicates
    /// the waveform has finished and will not generate any more samples. The value of any sample in `out`
    /// at or after the returned length is undefined.
    ///
    /// Waveforms should be initialized with `initialize_state` before the first call to `generate`. The
    /// returned waveform supports being "re-initialized" -- it will generate the same samples each time
    /// it's initialized.
    pub fn generate<M: Debug + Display>(
        &mut self,
        waveform: &mut Waveform<M>,
        out: &mut [f32],
    ) -> usize {
        use State::*;
        use waveform::Waveform::*;
        if out.len() == 0 {
            return 0;
        }
        match waveform {
            Const(value) => {
                out.fill(*value);
                out.len()
            }
            Time(state @ Initial) => {
                *state = Position(0);
                self.generate(waveform, out)
            }
            Time(Position(position)) => {
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (*position + i) as f32 / self.sample_rate as f32;
                }
                *position = *position + out.len();
                out.len()
            }
            Time(_) => unreachable!("Time waveform has non-Position state"),
            Noise => {
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                out.len()
            }
            Fixed(_, state @ Initial) => {
                *state = Position(0);
                self.generate(waveform, out)
            }
            Fixed(samples, Position(position)) => {
                if *position >= samples.len() {
                    return 0;
                }
                let len = (samples.len() - *position).min(out.len());
                out[..len].copy_from_slice(&samples[*position..(*position + len)]);
                *position = *position + len;
                len
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
                let len = self.length(waveform, out.len());
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
                self.generate(inner, &mut out[..len])
            }
            Append(a, b) => {
                let a_len = self.generate(a, out);
                if a_len == out.len() {
                    return a_len;
                }
                // a_len < out.len()
                let b_len = self.generate(b, &mut out[a_len..]);
                a_len + b_len
            }
            Sine {
                state: state @ Initial,
                ..
            } => {
                *state = Phase { accumulator: 0.0 };
                self.generate(waveform, out)
            }
            Sine {
                frequency,
                phase,
                state: Phase { accumulator },
            } => {
                // TODO Try to figure out if one the sub-waveforms is const and use
                // `out` for the other one.
                // Instantaneous frequency.
                let f_len = self.generate(frequency, out);
                // Instantaneous phase offset.
                let mut ph_out = vec![0.0; f_len];
                self.allocations += f_len;
                let ph_len = self.generate(phase, &mut ph_out);
                for (i, &phase_offset) in ph_out.iter().enumerate() {
                    let sample = (*accumulator + phase_offset as f64).sin() as f32;
                    let f = out[i] as f64;
                    let phase_inc = f / self.sample_rate as f64;
                    // Overwrite the frequency with the output.
                    out[i] = sample;
                    // Move the accumulator according to the frequency and change in phase offset.
                    *accumulator = (*accumulator + phase_inc).rem_euclid(f64::consts::TAU);
                }
                ph_len
            }
            Sine { .. } => unreachable!("Sine waveform has non-Initial, non-Phase state"),
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: state @ Initial,
            } => {
                // Generate the input samples so that they can be used in the feed-forward part of the filter.
                // TODO we could reverse the order of elements in input and output so that they line up with
                // how they are used.
                let ff_count = feed_forward.len();
                assert!(ff_count >= 1);
                let mut input = vec![0.0; ff_count - 1];
                self.allocations += ff_count - 1;
                let inner_len = self.generate(inner, &mut input);
                // Remove any unset elements from input.
                input.truncate(inner_len);
                let input = VecDeque::from(input);
                // Fill the previous output samples with zeros to match the number of feedback coefficients.
                let fb_count = feedback.len();
                let output = VecDeque::from(vec![0.0; fb_count]);
                self.allocations += fb_count;
                /*
                println!(
                    "Started generate for Filter with {} feed-forward and {} feedback coefficients",
                    ff_count, fb_count
                );
                */
                *state = Samples { input, output };
                self.generate(waveform, out)
            }
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: Samples { input, output },
            } => {
                // Overall, we want the length of the output to match that of the input `inner`. Doing so
                // means that the last output point will only depend on one input point (and the second to
                // last output will only depend on two input points, all the way back to the `ff_count-1`^th
                // to last point.) This means that if the inner waveform is not able to generate `out.len()`
                // points, we can zero-extend `input` up to `ff_count-1` points. We keep track of this by
                // saving fewer than ff_count points for the next call to `generate()`. This is the same
                // thing that happens in the case where the first call to `generate()` with Initial state
                // doesn't generate `ff_count-1` points.

                // First, generate samples from the inner waveform, if possible.
                let inner_len = self.generate(inner, out);
                let out_len = out.len().min(inner_len + input.len());
                // We may read past the end of inner_len, so make sure those samples are initialized.
                let extra_samples_read = out.len() - inner_len;
                out[inner_len..inner_len + extra_samples_read].fill(0.0);

                let ff_count = feed_forward.len();
                let input_padding = if input.len() == ff_count - 1 {
                    0
                } else {
                    // We've already exhausted `inner` on a previous call to generate (or in the Initial
                    // arm above), but we will be able to generate `input.len()` more samples. To do that,
                    // we need to pad up to `ff_count - 1`.
                    assert_eq!(0, inner_len);
                    (ff_count - 1) - input.len()
                };
                // Add the padding.
                input.resize(input.len() + input_padding, 0.0);
                assert_eq!(ff_count - 1, input.len());

                // The saved output should already be the correct size.
                let fb_count = feedback.len();
                assert_eq!(fb_count, output.len());

                // Set up the coefficients.
                // TODO I think this needs some optimization
                let mut recompute_coefficients = true;
                let mut ff_outs = vec![];
                let mut fb_outs = vec![];

                // Run the filter!!
                for x in out[..out_len].iter_mut() {
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
                            let mut ff_out = vec![0.0; 1];
                            self.allocations += 1;
                            _ = self.generate(w, &mut ff_out);
                            ff_outs.push(ff_out[0]);
                        }
                        for w in feedback.iter_mut() {
                            if let Const(_) = w {
                            } else {
                                recompute_coefficients = true;
                            }
                            let mut fb_out = vec![0.0; 1];
                            self.allocations += 1;
                            _ = self.generate(w, &mut fb_out);
                            fb_outs.push(fb_out[0]);
                        }
                        /*
                        println!(
                            "In generate() for Filter, generated coefficients:\n  b = {:?}\n  a = {:?}",
                            ff_outs, fb_outs
                        );
                        */
                    }

                    // First, save the current input value for future iterations.
                    input.push_back(*x);
                    // Since there's always at least one feed-forward coefficient, do that in-place.
                    *x *= ff_outs[0];
                    for (j, &ff) in ff_outs[1..].iter().enumerate() {
                        *x += ff * input[(ff_count - 1) - (j + 1)];
                    }
                    for (j, &fb) in fb_outs.iter().enumerate() {
                        *x -= fb * output[(fb_count - 1) - j];
                    }
                    // Discard the oldest input and output values.
                    input.pop_front();
                    // Push first into `output` in case `fb_count` is zero.
                    output.push_back(*x);
                    output.pop_front();
                }

                // `input` and `output` still contain the last few samples (`ff_count - 1` and
                // `fb_count` respectively). But if `input` was padded with any zeros or if we
                // copied any zeros from `out`, we don't want to save those.
                input.truncate(input.len() - (input_padding + extra_samples_read));
                out_len
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
                self.generate_binary_op(op_fn, a, b, *op == Operator::Merge, out)
            }
            Reset {
                state: state @ Initial,
                ..
            } => {
                // Start assuming that the trigger was previously negative.
                *state = Sign { signum: -1.0 };
                self.generate(waveform, out)
            }
            Reset {
                trigger,
                waveform: inner,
                state: Sign { signum },
            } => {
                let t_len = self.generate(trigger, out);

                // Keep track of how many samples we've generated so far.
                let mut generated = 0;
                while generated < t_len {
                    // Set to true if a restart will be triggered before the end of `out` is reached
                    let mut reset_inner_position = false;
                    let mut inner_desired = t_len - generated;

                    for (i, &x) in out[generated..].iter().enumerate() {
                        if *signum < 0.0 && x >= 0.0 {
                            inner_desired = i;
                            reset_inner_position = true;
                            *signum = x.signum();
                            break;
                        } else if *signum >= 0.0 && x < 0.0 {
                            *signum = x.signum();
                        }
                    }
                    // Overwrite the trigger values up to inner_desired.
                    let inner_len =
                        self.generate(inner, &mut out[generated..generated + inner_desired]);
                    // If the inner waveform ended early, fill with zeros.
                    out[generated + inner_len..generated + inner_desired].fill(0.0);
                    if reset_inner_position {
                        waveform::set_state(inner, State::Initial);
                    }
                    generated += inner_desired;
                }
                t_len
            }
            Reset { .. } => unreachable!("Reset waveform has non-Initial, non-Sign state"),
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let t_len = self.generate(trigger, out);
                // TODO we could avoid this allocation by generating runs from one of these...
                // though maybe we also need to advance them (is length() good enough?)
                // TODO maybe consider consts here specially?
                let mut positive_out = vec![0.0; t_len];
                self.allocations += t_len;
                let _ = self.generate(positive_waveform, &mut positive_out);
                let mut negative_out = vec![0.0; t_len];
                self.allocations += t_len;
                let _ = self.generate(negative_waveform, &mut negative_out);
                for (i, x) in out[..t_len].iter_mut().enumerate() {
                    if *x >= 0.0 {
                        *x = positive_out[i];
                    } else {
                        *x = negative_out[i];
                    }
                }
                t_len
            }
            Marked {
                waveform: inner, ..
            } => self.generate(inner, out),
            Captured {
                file_stem,
                waveform: inner,
            } => {
                // TODO think through this again
                //  - capture_state was set incorrectly when advancing position (i.e., when a waveform missed its start time)
                //  - we used to not generate the inner waveform when that was set... or was it unset?
                let len = self.generate(inner, out);
                if self.capture_state.is_none() {
                    // This occurs, for example, when precomputing parts of a waveform.
                    return len;
                }
                match self
                    .capture_state
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .get_mut(file_stem)
                {
                    Some(writer) => {
                        for x in out[..len].iter() {
                            if let Err(e) = writer.write_sample(*x) {
                                eprintln!("Error writing sample for {}: {}", file_stem, e);
                            }
                        }
                    }
                    None => {
                        panic!("No open file for captured waveform {}", file_stem);
                    }
                }
                len
            }
        }
    }

    // Generate a binary operation on two waveforms, up to 'desired' samples. The `op_fn` function
    // is applied to each pair of samples. If `extend_to_longer` is true, the shorter waveform will
    // be extended with zeros to match the length of the longer one.
    fn generate_binary_op<M: Debug + Display>(
        &mut self,
        op_fn: fn(f32, f32) -> f32,
        a: &mut Waveform<M>,
        b: &mut Waveform<M>,
        extend_to_longer: bool,
        out: &mut [f32],
    ) -> usize {
        // This approach may generate lots of samples from `a` that we don't need (in the case that `b`
        // is much shorter.)
        let a_len = self.generate(a, out);
        let len = if extend_to_longer { out.len() } else { a_len };
        match b {
            // Check to see if we can avoid generating the right-hand side and instead just apply the
            // op directly.
            waveform::Waveform::Const(f) => {
                // Make sure that any element where we apply `op` is initialized.
                out[a_len..len].fill(0.0);
                for x in out[..len].iter_mut() {
                    *x = op_fn(*x, *f);
                }
                // In theory, we need to advance `b`, but that's a no-op given that it's a Const
                len
            }
            _ => {
                let mut b_out = vec![0.0; len];
                self.allocations += len;
                let b_len = self.generate(b, &mut b_out);
                // Now we know both lengths, so determine the output length.
                let len = if extend_to_longer {
                    a_len.max(b_len)
                } else {
                    a_len.min(b_len)
                };
                // Make sure that any element where we apply `op` is initialized.
                out[a_len..len].fill(0.0);
                for (i, x) in out[..len].iter_mut().enumerate() {
                    *x = op_fn(*x, b_out[i]);
                }
                len
            }
        }
    }

    // Returns the number of samples that `waveform` will generate or `max`, whichever is
    // smaller, while modifying `waveform` so that it has been advanced to that point.
    // Note that for waveforms that maintain state (other than position), the state is passed
    // through unchanged. This discontinuity may result in pops or clicks for audible waveforms.
    pub fn length<M: Debug + Display>(&mut self, waveform: &mut Waveform<M>, max: usize) -> usize {
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
                        let mut length_out = vec![0.0; 1];
                        self.allocations += 1;
                        for i in 0..max {
                            let length_len = self.generate(length, &mut length_out);
                            if length_len == 0 {
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
                    input: VecDeque::from(vec![0.0; ff_count - 1]),
                    output: VecDeque::from(vec![0.0; fb_count]),
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
    fn greater_or_equals_at<M: Display>(
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

    /// Replaces parts of `waveform` that can be precomputed with their equivalent Fixed versions.
    ///
    /// Notably, infinite waveforms and waveforms that depend on or have dynamic behavior (Marked,
    /// Captured) cannot be replaced.
    pub fn precompute<M>(&mut self, waveform: waveform::Waveform<M>) -> waveform::Waveform<M>
    where
        M: Clone + Debug + Display,
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
        fn generate_fixed<M>(g: &mut Generator, mut waveform: Waveform<M>) -> Waveform<M>
        where
            M: Debug + Display,
        {
            // Choose a `desired` which is long enough to generate any reasonable waveform, but give some room
            // for cases like `Filter` that may need to make it longer.

            println!("Precomputing output for {}", &waveform);
            // XXX max precompute length?
            let max_len = (g.sample_rate * 10) as usize;
            let mut out = vec![0.0; max_len];
            let len = g.generate(&mut waveform, &mut out);
            if len == max_len {
                println!("Warning: precompute generated max samples (maybe not finite?)");
            }
            out.truncate(len);

            println!("  ...generated {} samples", out.len());

            waveform::Waveform::Fixed(out, State::Initial)
        }

        fn precompute_internal<M>(g: &mut Generator, waveform: Waveform<M>) -> Result<M>
        where
            M: Debug + Display,
        {
            use Reason::*;
            use Result::*;
            use waveform::Operator;
            use waveform::Waveform::*;

            // do_one_dynamic takes a waveform and determines if it can be pre-computed. If so, it applies `wf` and
            // then wraps the result in NPC(Dynamic).
            fn do_one_dynamic<M, F: FnOnce(Waveform<M>) -> Waveform<M>>(
                g: &mut Generator,
                a: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + Display,
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
                g: &mut Generator,
                a: Waveform<M>,
                b: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + Display,
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
                g: &mut Generator,
                a: Waveform<M>,
                b: Waveform<M>,
                c: Waveform<M>,
                wf: F,
            ) -> Result<M>
            where
                M: Debug + Display,
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
                    let mut extract = |r: Result<M>| match (r, &reason) {
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

        let result = match precompute_internal(self, initialize_state(waveform)) {
            Result::PC(w) => generate_fixed(self, w),
            Result::NPC(_, w) => w,
        };
        /*
        println!(
            "precompute waveform:\n{:#?}\ninto:\n{:#?}",
            &waveform, &result
        );
        */
        waveform::remove_state(result)
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
        g: &mut Generator,
        waveform: &Waveform,
        position: usize,
        expected: usize,
        max: usize,
    ) {
        let mut waveform = initialize_state(waveform.clone());
        // Generate up to `position` and discard those samples.
        let mut out = vec![0.0; position];
        let _ = g.generate(&mut waveform, &mut out);
        assert_eq!(
            g.length(&mut waveform.clone(), max),
            expected,
            "Expected length {} (with max = {}) for waveform: {:?}",
            expected,
            max,
            waveform
        );
    }

    fn run_tests(waveform: &Waveform, expected: &[f32]) {
        let mut g = new_test_generator(1);
        // We initialize output vectors with infinity to make sure that nothing depends on
        // them being initialized to 0.0.

        // Check that `waveform` would generate at least as many samples as `expected`.
        check_length(&mut g, &waveform, 0, expected.len(), expected.len());
        for size in [1, 2, 4, 8] {
            let mut w = initialize_state(waveform.clone());
            let mut out = vec![f32::INFINITY; expected.len()];
            for n in 0..out.len() / size + 1 {
                let end = out.len().min((n + 1) * size);
                let len = g.generate(&mut w, &mut out[n * size..end]);
                assert_eq!(end - n * size, len);
            }
            assert_eq!(
                out, *expected,
                "Failed output for size {} on waveform\n{:#?}",
                size, waveform
            );
        }

        let optimized_waveform = optimizer::optimize(waveform.clone());
        check_length(
            &mut g,
            &optimized_waveform,
            0,
            expected.len(),
            expected.len(),
        );
        for size in [1, 2, 4, 8] {
            let mut w = initialize_state(optimized_waveform.clone());
            let mut out = vec![f32::INFINITY; expected.len()];
            for n in 0..out.len() / size + 1 {
                let end = out.len().min((n + 1) * size);
                let len = g.generate(&mut w, &mut out[n * size..end]);
                assert_eq!(end - n * size, len);
            }
            assert_eq!(
                out, *expected,
                "Failed output for size {} on waveform\n{:#?}\noptimized to\n{:#?}",
                size, waveform, optimized_waveform
            );
        }

        let precomputed_waveform = g.precompute(optimized_waveform);
        check_length(
            &mut g,
            &precomputed_waveform,
            0,
            expected.len(),
            expected.len(),
        );
        for size in [1, 2, 4, 8] {
            let mut w = initialize_state(precomputed_waveform.clone());
            let mut out = vec![f32::INFINITY; expected.len()];
            for n in 0..out.len() / size + 1 {
                let end = out.len().min((n + 1) * size);
                let len = g.generate(&mut w, &mut out[n * size..end]);
                assert_eq!(end - n * size, len);
            }
            assert_eq!(
                out, *expected,
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

        let mut g = new_test_generator(1);
        let mut w = initialize_state(w);
        // Advance past the end of the waveform
        let mut out = vec![0.0; 6];
        let _ = g.generate(&mut w, &mut out);
        out.fill(0.0);
        let len = g.generate(&mut w, &mut out);
        assert_eq!(len, 0);
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

    fn run_sin_test<M>(g: &mut Generator, waveform: &mut super::Waveform<M>, expected: Vec<f32>)
    where
        M: Debug + Display,
    {
        let mut out = vec![0.0; expected.len()];
        let _ = g.generate(waveform, &mut out);
        for (i, &x) in out.iter().enumerate() {
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
    fn test_sine() {
        let sample_frequency = 44100.0;
        let mut g = new_test_generator(sample_frequency as i32);

        // As simple as possible
        let mut w = initialize_state(*sin_waveform(1.0, 0.0));
        let expected = (0..100)
            .map(|x: i32| (f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&mut g, &mut w, expected);

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
        run_sin_test(&mut g, &mut w, expected);

        // Non-zero phase offset
        let mut w = initialize_state(*sin_waveform(0.25, f32::consts::PI));
        let expected = (0..100)
            .map(|x| {
                (f64::consts::TAU * 0.25 * x as f64 / sample_frequency + f64::consts::PI).sin()
                    as f32
            })
            .collect();
        run_sin_test(&mut g, &mut w, expected);
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
        let mut g = new_test_generator(1);
        let w = Append(
            Box::new(Fixed(vec![1.0; 3], ())),
            Box::new(Fixed(vec![2.0; 3], ())),
        );

        check_length(&mut g, &w, 0, 6, MAX_LENGTH);
        check_length(&mut g, &w, 2, 4, MAX_LENGTH);
        check_length(&mut g, &w, 4, 2, MAX_LENGTH);
        run_tests(&w, &vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        match g.precompute(w) {
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
        let mut g = new_test_generator(1);

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
        match g.precompute(w) {
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
        let mut g = new_test_generator(1);

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
        check_length(&mut g, &w, 0, 3, 5);

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
        let mut g = new_test_generator(1);
        check_length(&mut g, &w, 0, 8, MAX_LENGTH);
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
        let mut g = new_test_generator(1);
        let position = g.greater_or_equals_at(&initialize_state(w1.clone()), 0.0, 10);
        let mut out = vec![0.0; 10];
        let len = g.generate(&mut initialize_state(w2), &mut out);
        assert!(position.is_some());
        assert_eq!(position.unwrap(), len);
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
