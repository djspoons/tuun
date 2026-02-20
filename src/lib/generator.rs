use std::cell::RefCell;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::io::BufWriter;

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
    // Previously generated samples as input and/or output to the waveform. (for example, for Filter)
    Samples {
        input: Vec<f32>,
        output: Vec<f32>,
    },
    // The current accumulated phase as well the previous value of the frequency waveform (for example, for Sin).
    Phase {
        accumulator: f64,
        previous_frequency: f64,
    },
    // The sign of the last generated value along with the direction of generation (for example, for Res).
    Sign {
        signum: f32,
    },
}

pub fn initialize_state<S>(waveform: waveform::Waveform<S>) -> Waveform {
    return waveform::initialize_state(waveform, State::Initial);
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

pub type Waveform = waveform::Waveform<State>;

// TODO add metrics for waveform expr depth and total ops

#[derive(Debug, Copy, Clone)]
enum MaybeOption<T> {
    Some(T),
    None,
    Maybe, // The value may or may not be present
}

const MAX_FILTER_LENGTH: usize = 1000; // Maximum length of the filter coefficients

fn truncate_or_pad_left<T>(mut v: Vec<T>, size: usize, val: T) -> Vec<T>
where
    T: Clone,
{
    if v.len() == size {
        return v;
    } else if v.len() > size {
        return v.drain(v.len() - size..).collect();
    } else {
        // v.len() < size
        let mut out = vec![val; size - v.len()];
        out.append(&mut v);
        return out;
    }
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

    // Generate a vector of samples up to `desired` length and return a new waveform that will continue where this
    // one left off. If fewer than 'desired' samples are generated, that indicates that this waveform has finished.
    // Waveforms should be initialized with `initialize_state` before the first call to `generate`. The returned
    // waveform supports being "re-initialized" -- unless it depends on some external state (as Slider does), it
    // will generate the same samples each time it's initialized.
    pub fn generate(&self, waveform: Waveform, desired: usize) -> (Waveform, Vec<f32>) {
        use State::*;
        use waveform::Waveform::*;
        if desired == 0 {
            return (waveform, vec![]);
        }
        match waveform {
            Const(value) => {
                return (waveform, vec![value; desired]);
            }
            Time(Initial) => self.generate(Time(Position(0)), desired),
            Time(Position(position)) => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (position + i) as f32 / self.sample_frequency as f32;
                }
                return (Time(Position(position + desired)), out);
            }
            Time(_) => unreachable!("Time waveform has non-Position state"),
            Noise => {
                let mut out = vec![0.0; desired];
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                return (waveform, out);
            }
            Fixed(samples, Initial) => self.generate(Fixed(samples, Position(0)), desired),
            Fixed(samples, Position(position)) => {
                let mut out = vec![];
                if position < samples.len() {
                    out.extend(&samples[position..(position + desired).min(samples.len())]);
                }
                return (Fixed(samples, Position(position + out.len())), out);
            }
            Fixed(_, _) => unreachable!("Fixed waveform has non-Initial, non-Position state"),
            Fin {
                length,
                waveform: inner,
            } => {
                // Note that this call to remaining also advances the position of the `length` waveform.
                if let (Fin { length, .. }, len) = self.remaining(
                    Fin {
                        length,
                        waveform: inner.clone(),
                    },
                    desired,
                ) {
                    let (inner, out) = self.generate(*inner, len);
                    return (
                        Fin {
                            length,
                            waveform: Box::new(inner),
                        },
                        out,
                    );
                } else {
                    unreachable!("remaining(Fin) returned a non-Fin waveform")
                }
            }
            // XXX Maybe remove this?
            Seq {
                offset,
                waveform: inner,
            } => {
                let (inner, out) = self.generate(*inner, desired);
                return (
                    Seq {
                        offset,
                        waveform: Box::new(inner),
                    },
                    out,
                );
            }
            Append(a, b) => {
                let (a, mut a_out) = self.generate(*a, desired);
                if a_out.len() == desired {
                    return (Append(Box::new(a), b), a_out);
                } else {
                    // a_out.len() < desired
                    let (b, mut b_out) = self.generate(*b, desired - a_out.len());
                    a_out.append(&mut b_out);
                    return (Append(Box::new(a), Box::new(b)), a_out);
                }
            }
            Sin {
                frequency,
                phase,
                state: Initial,
            } => {
                // Frequency is going to always be one position "ahead" of `waveform` so we generate one
                // value at the origin now.
                let (frequency, f_out) = self.generate(*frequency, 1);
                if f_out.is_empty() {
                    return (
                        Sin {
                            frequency: Box::new(frequency),
                            phase: Box::new(*phase),
                            state: Initial,
                        },
                        vec![],
                    );
                }
                return self.generate(
                    Sin {
                        frequency: Box::new(frequency),
                        phase: Box::new(*phase),
                        state: Phase {
                            accumulator: 0.0,
                            previous_frequency: f_out[0] as f64,
                        },
                    },
                    desired,
                );
            }
            Sin {
                frequency,
                phase,
                state:
                    Phase {
                        mut accumulator,
                        mut previous_frequency,
                    },
            } => {
                // Instantaneous frequency (note that this is one position ahead of `waveform`).
                let (frequency, f_out) = self.generate(*frequency, desired);
                // Instantaneous phase offset.
                let (phase, ph_out) = self.generate(*phase, f_out.len());
                let mut out = vec![0.0; ph_out.len()];
                for (i, &phase_offset) in ph_out.iter().enumerate() {
                    out[i] = (accumulator + phase_offset as f64).sin() as f32;

                    // Take the average of the last two frequency values (i.e., the trapezoidal
                    // approximation to the integral).
                    let f = (previous_frequency + f_out[i] as f64) / 2.0;
                    let phase_inc = f / self.sample_frequency as f64;

                    // Move the accumulator according to the frequency and change in phase offset.
                    accumulator = (accumulator + phase_inc).rem_euclid(f64::consts::TAU);
                    previous_frequency = f_out[i] as f64;
                }
                return (
                    Sin {
                        frequency: Box::new(frequency),
                        phase: Box::new(phase),
                        state: Phase {
                            accumulator,
                            previous_frequency,
                        },
                    },
                    out,
                );
            }
            Sin { .. } => unreachable!("Sin waveform has non-Initial, non-Phase state"),
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state: Initial,
            } => {
                let (_, ff_len) = self.remaining(*feed_forward.clone(), MAX_FILTER_LENGTH);
                // Generate the input samples so that they can be used in the feed forward part of the filter. Note
                // that we could do this later but that complicates the management of the inner waveform.
                let (inner, input) = self.generate(*inner, ff_len - 1);
                let (_, fb_len) = self.remaining(*feedback.clone(), MAX_FILTER_LENGTH);
                // Fill the previous output samples with zeros to match the length of `feedback`.
                let output = vec![0.0; fb_len];
                /*
                println!(
                    "Started generate for Filter({:?}, {:?}, {:?}) with {} samples for feed-forward and {} samples for feedback",
                    inner, feed_forward, feedback, ff_len, fb_len
                );
                */
                return self.generate(
                    Filter {
                        waveform: Box::new(inner),
                        feed_forward,
                        feedback,
                        state: Samples { input, output },
                    },
                    desired,
                );
            }
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state:
                    Samples {
                        input: previous_in,
                        output: previous_out,
                    },
            } => {
                // Generate the filter coefficients.
                // TODO should we rationalize how often the parameters are evaluated? Should it be on every sample? Or
                // at least something more predictable than once per buffer?
                let (_, ff_out) =
                    self.generate(initialize_state(*feed_forward.clone()), MAX_FILTER_LENGTH);
                let (_, fb_out) =
                    self.generate(initialize_state(*feedback.clone()), MAX_FILTER_LENGTH);

                // Set up the input and output. Each will have extra samples at the beginning from a previous
                // call to `generate`.
                let mut input = truncate_or_pad_left(previous_in, ff_out.len() - 1, 0.0);
                let (inner, mut inner_out) = self.generate(*inner, desired);
                input.append(&mut inner_out);
                // Use `previous_out` to fill in samples in `out` that we'll use to compute the feedback part of the
                // filter. We'll trim off these extra samples at the end.
                let mut out = truncate_or_pad_left(previous_out, fb_out.len(), 0.0);
                // Set the output length based on the size of the inner waveform.
                out.resize(input.len() - (ff_out.len() - 1) + fb_out.len(), 0.0);

                // Run the filter!!
                for i in fb_out.len()..out.len() {
                    for (j, &ff) in ff_out.iter().enumerate() {
                        out[i] += ff * input[i - fb_out.len() + (ff_out.len() - 1) - j];
                    }
                    for (j, &fb) in fb_out.iter().enumerate() {
                        out[i] -= fb * out[i - j - 1];
                    }
                }

                // Save the last few samples of both the input and the output.
                let previous_in = input[input.len() - (ff_out.len() - 1)..].to_vec();
                let previous_out = out[out.len() - fb_out.len()..].to_vec();

                // Remove the fb_out.len() samples from the beginning of `out` (these were included to compute
                // the feedback part of the filter).
                out.drain(0..fb_out.len());
                return (
                    Filter {
                        waveform: Box::new(inner),
                        feed_forward,
                        feedback,
                        state: Samples {
                            input: previous_in,
                            output: previous_out,
                        },
                    },
                    out,
                );
            }
            Filter { .. } => unreachable!("Filter waveform has non-Initial, non-Samples state"),
            BinaryPointOp(op, a, b) => {
                use waveform::Operator;
                let desired = match op {
                    Operator::Multiply | Operator::Divide => {
                        // We need to make sure we generate a length based on the shorter waveform.
                        // TODO would it be better to push this down into generate_binary_op?
                        let (_, len) = self
                            .remaining(BinaryPointOp(op.clone(), a.clone(), b.clone()), desired);
                        len
                    }
                    _ => desired,
                };
                let op_fn = match op {
                    Operator::Add => std::ops::Add::add,
                    Operator::Subtract => std::ops::Sub::sub,
                    Operator::Multiply => std::ops::Mul::mul,
                    Operator::Divide => |a: f32, b: f32| {
                        if b == 0.0 { 0.0 } else { a / b }
                    },
                };
                return self.generate_binary_op(op, op_fn, *a, *b, desired);
            }
            Res {
                trigger,
                waveform,
                state: Initial,
            } => {
                // Start assuming that the trigger was previously negative.
                return self.generate(
                    Res {
                        trigger,
                        waveform,
                        state: Sign { signum: -1.0 },
                    },
                    desired,
                );
            }
            Res {
                trigger,
                waveform: inner,
                state: Sign { mut signum },
            } => {
                let mut generated = 0;
                let mut out = Vec::new();
                let mut inner = *inner;

                let (trigger, t_out) = self.generate(*trigger, desired);

                while generated < t_out.len() {
                    // Set to true if a restart will be triggered before desired is reached
                    let mut reset_inner_position = false;
                    let mut inner_desired = t_out.len() - generated;

                    for (i, &x) in t_out[generated..].iter().enumerate() {
                        if signum < 0.0 && x >= 0.0 {
                            inner_desired = i;
                            reset_inner_position = true;
                            signum = x.signum();
                            break;
                        } else if signum >= 0.0 && x < 0.0 {
                            signum = x.signum();
                        }
                    }

                    let (new_inner, mut tmp) = self.generate(inner, inner_desired);
                    inner = new_inner;
                    if tmp.len() < inner_desired {
                        tmp.resize(inner_desired, 0.0);
                    }
                    out.extend(tmp);
                    if reset_inner_position {
                        inner = initialize_state(inner);
                    }
                    generated += inner_desired;
                }
                return (
                    Res {
                        trigger: Box::new(trigger),
                        waveform: Box::new(inner),
                        state: Sign { signum },
                    },
                    out,
                );
            }
            Res { .. } => unreachable!("Res waveform has non-Initial, non-Sign state"),
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let (trigger, mut out) = self.generate(*trigger, desired);
                let (positive_waveform, mut positive_out) =
                    self.generate(*positive_waveform, desired);
                positive_out.resize(out.len(), 0.0);
                let (negative_waveform, mut negative_out) =
                    self.generate(*negative_waveform, desired);
                negative_out.resize(out.len(), 0.0);
                for (i, x) in out.iter_mut().enumerate() {
                    if *x >= 0.0 {
                        *x = positive_out[i];
                    } else {
                        *x = negative_out[i];
                    }
                }
                return (
                    Alt {
                        trigger: Box::new(trigger),
                        positive_waveform: Box::new(positive_waveform),
                        negative_waveform: Box::new(negative_waveform),
                    },
                    out,
                );
            }
            Slider(slider) => {
                if self.slider_state.is_none() {
                    println!("Warning: Slider waveform used, but no slider state set");
                    return (Slider(slider), vec![0.0; desired]);
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
                return (Slider(slider), out);
            }
            Marked {
                waveform: inner,
                id,
            } => {
                let (inner, out) = self.generate(*inner, desired);
                return (
                    Marked {
                        waveform: Box::new(inner),
                        id,
                    },
                    out,
                );
            }
            Captured {
                file_stem,
                waveform: inner,
            } => {
                // TODO think through this again
                //  - capture_state was set incorrectly when advancing position (i.e., when a waveform missed its start time)
                //  - we used to not generate the inner waveform when that was set... or was it unset?
                let (inner, out) = self.generate(*inner, desired);
                if self.capture_state.is_none() {
                    // This occurs, for example, when precomputing parts of a waveform.
                    return (
                        Captured {
                            file_stem,
                            waveform: Box::new(inner),
                        },
                        out,
                    );
                }
                match self
                    .capture_state
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .get_mut(&file_stem)
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
                return (
                    Captured {
                        file_stem,
                        waveform: Box::new(inner),
                    },
                    out,
                );
            }
        }
    }

    // Generate a binary operation on two waveforms, up to 'desired' samples. The second waveform is
    // offset by the offset of the first waveform. The `op` function is applied to each pair of samples.
    fn generate_binary_op(
        &self,
        op: waveform::Operator,
        op_fn: fn(f32, f32) -> f32,
        a: Waveform,
        b: Waveform,
        desired: usize,
    ) -> (Waveform, Vec<f32>) {
        use waveform::Waveform::BinaryPointOp;
        assert_eq!(self.offset(&a, desired), 0);
        let (a, mut a_out) = self.generate(a, desired);
        match b {
            // In this branch (which is always taken if we've removed Seq's), check to see if we can
            // avoid generating the right-hand side and instead just apply the op directly.
            // TODO could also check f against the identity of op and skip the loop here
            Waveform::Const(f) => {
                for x in a_out.iter_mut() {
                    *x = op_fn(*x, f);
                }
                if desired > a_out.len() {
                    a_out.resize(desired, op_fn(0.0, f));
                }
                // In theory, we need to advance `b`, but that's a no-op given that it's a Const
                (BinaryPointOp(op, Box::new(a), Box::new(b)), a_out)
            }
            _ => {
                let (b, b_out) = self.generate(b, desired);
                // Merge the overlapping portion
                for (i, x) in a_out.iter_mut().enumerate() {
                    if i >= b_out.len() {
                        break;
                    }
                    *x = op_fn(*x, b_out[i]);
                }
                // If the left side is shorter than the right, than append.
                if b_out.len() > a_out.len() {
                    a_out.extend_from_slice(&b_out[a_out.len()..]);
                }
                (BinaryPointOp(op, Box::new(a), Box::new(b)), a_out)
            }
        }
    }

    // Returns the number of samples that `waveform` will generate or `max`, whichever is
    // smaller, along with a copy of `waveform` that has been advanced to that point.
    // Note that for waveforms that maintain state (other than position), the state is passed
    // through unchanged. This discontinuity may result in pops or clicks for audible waveforms.
    // XXXXX rename as 'length'?
    pub fn remaining(&self, waveform: Waveform, max: usize) -> (Waveform, usize) {
        use State::*;
        use waveform::Operator;
        use waveform::Waveform::*;
        match waveform {
            Const { .. } => (waveform, max),
            Time(Initial) => self.remaining(Time(Position(0)), max),
            Time(Position(position)) => (Time(Position(position + max)), max),
            Time(_) => unreachable!("Time waveform with non-Initial, non-Position state"),
            Noise => (Noise, max),
            Fixed(samples, Initial) => self.remaining(Fixed(samples, Position(0)), max),
            Fixed(samples, Position(position)) => {
                if position >= samples.len() {
                    return (Fixed(samples, Position(position)), 0);
                }
                let len = max.min(samples.len() - position);
                return (Fixed(samples, Position(position + len)), len);
            }
            Fixed(_, _) => unreachable!("Fixed waveform with non-Initial, non-Position state"),
            Fin {
                length,
                waveform: inner,
            } => {
                // This is a little subtle, since Fin is not supposed to make waveforms longer. In particular, for
                // optimizations that move Fin outside of a DotProduct, we need to check the length of the inner
                // waveform to see if it's shorter than `length` would indicate.
                // TODO figure out which of `length` or `length(waveform)` is cheaper and do that first.
                let (new_inner, inner_len) = self.remaining(*inner.clone(), max);
                match self.greater_or_equals_at(&length, 0.0, inner_len) {
                    MaybeOption::Some(len) => {
                        // Advance `length` by `len` samples and make sure that `inner` is at the right point too
                        let (length, _) = self.remaining(*length, len);
                        let (inner, _) = self.remaining(*inner, len);
                        return (
                            Fin {
                                length: Box::new(length),
                                waveform: Box::new(inner),
                            },
                            len,
                        );
                    }
                    MaybeOption::None => {
                        // Advance `length` by `inner_len` samples
                        let (length, _) = self.remaining(*length, inner_len);
                        return (
                            Fin {
                                length: Box::new(length),
                                waveform: Box::new(new_inner),
                            },
                            inner_len,
                        );
                    }
                    MaybeOption::Maybe => {
                        println!(
                            "Warning: unable to determine root of Fin length cheaply, generating samples for: {:?}",
                            length
                        );
                        let (new_length, length_out) = self.generate(*length.clone(), inner_len);
                        for (i, &x) in length_out.iter().enumerate() {
                            if x >= 0.0 {
                                // Advance the original `length` and `inner` by `i` samples
                                let (length, _) = self.remaining(*length, i);
                                let (inner, _) = self.remaining(*inner, i);
                                return (
                                    Fin {
                                        length: Box::new(length),
                                        waveform: Box::new(inner),
                                    },
                                    i,
                                );
                            }
                        }
                        // Note that `new_length` and `new_inner` are already advanced by `inner_len` samples
                        return (
                            Fin {
                                length: Box::new(new_length),
                                waveform: Box::new(new_inner),
                            },
                            inner_len,
                        );
                    }
                }
            }
            Seq {
                offset,
                waveform: inner,
            } => {
                let (inner, len) = self.remaining(*inner, max);
                return (
                    Seq {
                        offset,
                        waveform: Box::new(inner),
                    },
                    len,
                );
            }
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state,
            } => {
                let (_, ff_len) = self.remaining(initialize_state(*feed_forward.clone()), max);
                let (inner, inner_len) = self.remaining(*inner, max + (ff_len.saturating_sub(1)));
                return (
                    Filter {
                        waveform: Box::new(inner),
                        feed_forward,
                        feedback,
                        state,
                    },
                    inner_len.saturating_sub(ff_len.saturating_sub(1)),
                );
            }
            Append(a, b) => {
                let (a, a_len) = self.remaining(*a, max);
                let (b, b_len) = self.remaining(*b, max - a_len);
                (Append(Box::new(a), Box::new(b)), a_len + b_len)
            }
            Sin {
                frequency,
                phase,
                state,
            } => {
                let (frequency, f_len) = self.remaining(*frequency, max);
                let (phase, ph_len) = self.remaining(*phase, max);
                (
                    Sin {
                        frequency: Box::new(frequency),
                        phase: Box::new(phase),
                        state,
                    },
                    f_len.min(ph_len),
                )
            }
            BinaryPointOp(op, a, b) => {
                let (a, a_len) = self.remaining(*a, max);
                assert_eq!(self.offset(&a, max), 0); // XXXXX remove this?
                let (b, b_len) = self.remaining(*b, max);
                (
                    BinaryPointOp(op.clone(), Box::new(a), Box::new(b)),
                    match op {
                        Operator::Add | Operator::Subtract => a_len.max(b_len),
                        Operator::Multiply | Operator::Divide => a_len.min(b_len),
                    },
                )
            }
            Res {
                trigger,
                waveform,
                state,
            } => {
                let (trigger, len) = self.remaining(*trigger, max);
                (
                    Res {
                        trigger: Box::new(trigger),
                        waveform,
                        state,
                    },
                    len,
                )
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                let (trigger, len) = self.remaining(*trigger, max);
                (
                    Alt {
                        trigger: Box::new(trigger),
                        positive_waveform,
                        negative_waveform,
                    },
                    len,
                )
            }
            Slider(slider) => (Slider(slider), max),
            Marked { waveform, id } => {
                let (waveform, len) = self.remaining(*waveform, max);
                (
                    Marked {
                        waveform: Box::new(waveform),
                        id,
                    },
                    len,
                )
            }
            Captured {
                waveform,
                file_stem,
            } => {
                let (waveform, len) = self.remaining(*waveform, max);
                (
                    Captured {
                        waveform: Box::new(waveform),
                        file_stem,
                    },
                    len,
                )
            }
        }
    }

    // If `waveform` will be greater than or equal to `value` at some point between its current position and `max`,
    // return Some of the number of samples that would be generated before then, None if `waveform` will not be greater
    // than or equal in that range, or Maybe if that can't be determined cheaply.
    fn greater_or_equals_at(
        &self,
        waveform: &Waveform,
        value: f32,
        max: usize,
    ) -> MaybeOption<usize> {
        use State::{Initial, Position};
        use waveform::Operator;
        use waveform::Waveform::{Append, BinaryPointOp, Const, Time};
        match waveform {
            Const(v) if *v >= value => MaybeOption::Some(0),
            Const(_) => MaybeOption::None,
            Time(Initial) => self.greater_or_equals_at(&Time(Position(0)), value, max),
            Time(Position(position)) => {
                let current_value = *position as f32 / self.sample_frequency as f32;
                if current_value >= value {
                    MaybeOption::Some(0)
                } else {
                    // current_value < value and current_value >= 0 so value > 0 (so usize is ok)
                    let target_position = (value * self.sample_frequency as f32).ceil() as usize;
                    // Also, target_position must be > position
                    MaybeOption::Some(max.min((target_position - position) as usize))
                }
            }
            Append(a, b) => {
                match self.greater_or_equals_at(a, value, max) {
                    MaybeOption::Some(size) => MaybeOption::Some(size),
                    MaybeOption::None => {
                        let (_, a_len) = self.remaining(*a.clone(), max);
                        if a_len == max {
                            // We didn't reach the end of `a``, so `b` isn't relevant yet.
                            MaybeOption::None
                        } else {
                            match self.greater_or_equals_at(&b, value, max - a_len) {
                                MaybeOption::Some(size) => MaybeOption::Some(size + a_len),
                                m => m,
                            }
                        }
                    }
                    m => m, // Maybe gets passed through
                }
            }
            BinaryPointOp(op @ (Operator::Add | Operator::Subtract), a, b) => {
                use waveform::Operator::{Add, Subtract};
                // If a has an offset, we'd need to handle the overlapping and non-overlapping parts
                // separately, and since we don't expect this to be called on non-optimized waveforms (except
                // in tests), just give up.
                // TODO update this
                if self.offset(a, max) != 0 {
                    return MaybeOption::Maybe;
                }
                match (op, a.as_ref(), b.as_ref()) {
                    // TODO need to consider Sliders and constant functions of Sliders as const
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
            _ => MaybeOption::Maybe,
        }
    }

    // Returns the offset at which the next waveform should start (in samples). This is determined entirely by
    // `waveform` (this function is pure).
    // XXX Maybe move it to the test section?
    pub fn offset(&self, waveform: &Waveform, max: usize) -> usize {
        use waveform::Waveform::*;
        match waveform {
            Const { .. } | Time(_) | Noise | Fixed(_, _) => 0,
            Fin { waveform, .. } => self.offset(waveform, max),
            Seq { offset, .. } => match self.greater_or_equals_at(&offset, 0.0, max) {
                MaybeOption::Some(size) => size,
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

    // Replaces parts of `waveform` that can be precomputed with their equivalent Fixed versions. Notably,
    // infinite waveforms and waveforms that depend on or have dynamic behavior (Slider, Marked, Captured)
    // cannot be replaced. This should be called after replace_seq.
    pub fn precompute(&self, waveform: Waveform) -> Waveform {
        #[derive(Clone, Copy)]
        enum Reason {
            // The waveform is not pre-computable because it is infinite in length.
            Infinite,
            // The waveform is not pre-computable because it depends on some run-time state or has a run-time effect.
            Dynamic,
        }
        enum Result {
            // Pre-computable
            PC(Waveform),
            // Not-pre-computable
            NPC(Reason, Waveform),
        }

        impl Into<Waveform> for Result {
            fn into(self) -> Waveform {
                match self {
                    Result::PC(w) => w,
                    Result::NPC(_, w) => w,
                }
            }
        }

        // generate_fixed generates `waveform` up to some large number of samples. It should only be used on waveforms
        // that are pre-computable.
        fn generate_fixed(g: &Generator, waveform: Waveform) -> Waveform {
            // Choose a `desired` which is long enough to generate any reasonable waveform, but give some room
            // for cases like `Filter` that may need to make it longer.
            let (_, out) = g.generate(waveform, usize::MAX / 2);
            waveform::Waveform::Fixed(out, State::Initial)
        }

        fn precompute_internal(g: &Generator, waveform: Waveform) -> Result {
            use Reason::*;
            use Result::*;
            use waveform::Operator;
            use waveform::Waveform::*;

            // do_one_dynamic takes a waveform and determines if it can be pre-computed. If so, it applies `wf` and
            // then wraps the result in NPC(Dynamic).
            fn do_one_dynamic<F: FnOnce(Waveform) -> Waveform>(
                g: &Generator,
                a: Waveform,
                wf: F,
            ) -> Result {
                match precompute_internal(g, a) {
                    PC(a) => NPC(Dynamic, wf(generate_fixed(g, a))),
                    NPC(why, a) => NPC(why, wf(a)),
                }
            }

            // do_two attempts to pre-compute two waveforms, then applies `wf` to the results. If both waveforms are
            // pre-computable, then the result is wrapped in PC. If at least one is not pre-computable, then the result
            // is wrapped in NPC, with the reason determined by the reason(s) for the two waveforms.
            fn do_two<F: FnOnce(Waveform, Waveform) -> Waveform>(
                g: &Generator,
                a: Waveform,
                b: Waveform,
                wf: F,
            ) -> Result {
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
            fn do_three<F: FnOnce(Waveform, Waveform, Waveform) -> Waveform>(
                g: &Generator,
                a: Waveform,
                b: Waveform,
                c: Waveform,
                wf: F,
            ) -> Result {
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
                            "Cannot precompute Fin because inner waveform is dynamic: {:?}",
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
                            "Cannot precompute Fin because length waveform is dynamic: {:?}",
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
                Seq { .. } => {
                    panic!(
                        "Seq should have been replaced by replace_seq before precompute is called"
                    );
                }
                Append(a, b) => do_two(g, *a, *b, |a, b| Append(Box::new(a), Box::new(b))),
                Sin {
                    frequency,
                    phase,
                    state,
                } => do_two(g, *frequency, *phase, |frequency, phase| Sin {
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
                } => do_three(
                    g,
                    *waveform,
                    *feed_forward,
                    *feedback,
                    |waveform, feed_forward, feedback| Filter {
                        waveform: Box::new(waveform),
                        feed_forward: Box::new(feed_forward),
                        feedback: Box::new(feedback),
                        state,
                    },
                ),
                Res {
                    trigger,
                    waveform,
                    state,
                } => do_two(g, *trigger, *waveform, |trigger, waveform| Res {
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
                // Slider is always dynamic.
                Slider(_) => NPC(Dynamic, waveform),
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
                Box::new(Time(())),
                Box::new(Const(seq_duration as f32)),
            )),
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(fin_duration as f32)),
                )),
                waveform: Box::new(Const(value)),
            }),
        };
    }

    // Verifies that `waveform` has an offset of `expected` samples.
    fn check_offset(g: &Generator, waveform: &Waveform, expected: usize) {
        let (offset_waveform, _) = optimizer::replace_seq(waveform.clone());
        let offset_waveform = initialize_state(offset_waveform);
        match g.greater_or_equals_at(&offset_waveform, 0.0, MAX_LENGTH) {
            MaybeOption::Some(offset) => {
                assert_eq!(offset, expected);
            }
            m => {
                assert!(
                    false,
                    "Failed to check offset waveform {:?} (expected {}, got {:?})",
                    waveform, expected, m
                );
            }
        }
    }

    // Verifies that `waveform` would generate `expected` samples using a call to `remaining`.
    fn check_length(
        g: &Generator,
        waveform: &Waveform,
        position: usize,
        expected: usize,
        max: usize,
    ) {
        let (_, waveform) = optimizer::replace_seq(waveform.clone());
        let waveform = initialize_state(waveform);
        // Generate up to `position` and discard those samples.
        let (waveform, _) = g.generate(waveform, position);
        assert_eq!(
            g.remaining(waveform.clone(), max).1,
            expected,
            "Expected length {} (with max = {}) for waveform: {:?}",
            expected,
            max,
            waveform
        );
    }

    fn run_tests(waveform: &waveform::Waveform, desired: &Vec<f32>) {
        let g = new_test_generator(1);
        // Check that `waveform` would generate at least as many samples as `desired`.
        check_length(&g, &waveform, 0, desired.len(), desired.len());
        /*
        // We can't currently generate for waveforms that contain Seq, so skip this.
        for size in [1, 2, 4, 8] {
            let g = new_test_generator(1);
            let mut w = initialize_state(waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (w, tmp) = g.generate(w, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, desired,
                "Failed output for size {} on waveform:\n{:#?}",
                size, waveform
            );
        }
        */

        for size in [1, 2, 4, 8] {
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            check_length(&g, &no_seq_waveform, 0, desired.len(), desired.len());
            let mut w = initialize_state(no_seq_waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (new_w, tmp) = g.generate(w, size);
                w = new_w;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let (_, tmp) = g.generate(w, out.len() % size);
                let n = (out.len() - 1) / size;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(
                out, *desired,
                "Failed output for size {} on waveform\n{:#?}\nwith seq's removed\n{:#?}",
                size, waveform, no_seq_waveform
            );
        }

        for size in [1, 2, 4, 8] {
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            let optimized_waveform = optimizer::simplify(no_seq_waveform.clone());
            check_length(&g, &optimized_waveform, 0, desired.len(), desired.len());
            let mut w = initialize_state(optimized_waveform.clone());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (new_w, tmp) = g.generate(w, size);
                w = new_w;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let (_, tmp) = g.generate(w, out.len() % size);
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
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            let optimized_waveform = optimizer::simplify(no_seq_waveform.clone());
            let precomputed_waveform = g.precompute(initialize_state(optimized_waveform));
            let mut w = precomputed_waveform.clone();
            // XXX check_length(&g, &w, 0, desired.len(), desired.len());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (new_w, tmp) = g.generate(w, size);
                w = new_w;
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            if out.len() % size != 0 {
                // Generate any remaining samples
                let (_, tmp) = g.generate(w, out.len() % size);
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
        let w = initialize_state(w);
        // Advance past the end of the waveform
        let (w, _) = g.generate(w, 6);
        let (_, result) = g.generate(w, 8);
        assert_eq!(result, vec![] as Vec<f32>);
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

    fn run_sin_test(g: &Generator, waveform: &super::Waveform, expected: Vec<f32>) {
        let (_, result) = g.generate(waveform.clone(), expected.len());
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
        let sample_frequency = 100.0;
        let g = new_test_generator(sample_frequency as i32);

        // As simple as possible
        let w = initialize_state(*sin_waveform(1.0, 0.0));
        let expected = (0..100)
            .map(|x: i32| (f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&g, &w, expected);

        // Non-constant frequency: f = time + 10 Hz
        let w3 = initialize_state(Waveform::Sin {
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
        run_sin_test(&g, &w3, expected);

        // Non-zero phase offset
        let w = initialize_state(*sin_waveform(0.25, f32::consts::PI));
        let expected = (0..100)
            .map(|x| {
                (f64::consts::TAU * 0.25 * x as f64 / sample_frequency + f64::consts::PI).sin()
                    as f32
            })
            .collect();
        run_sin_test(&g, &w, expected);
    }

    #[test]
    fn test_res() {
        let w1 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w1, &vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        let w2 = Res {
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
        run_tests(&w2, &vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

        let w3 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Waveform::Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(3.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            state: (),
        };
        run_tests(&w3, &vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0]);

        let w5 = Res {
            trigger: sin_waveform(0.25, f32::consts::PI),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w5, &vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

        // Test where a reset lines up with the buffer boundary and where there are multiple
        // resets in a buffer.
        let w6 = Res {
            trigger: sin_waveform(0.25, 0.0),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(
            &w6,
            &vec![
                0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
            ],
        );
    }

    #[test]
    fn test_append() {
        let g = new_test_generator(1);
        let w = Append(
            Box::new(finite_const_waveform(1.0, 3, 2)),
            Box::new(finite_const_waveform(2.0, 3, 2)),
        );

        check_offset(&g, &w, 4);
        check_length(&g, &w, 0, 6, MAX_LENGTH);
        check_length(&g, &w, 2, 4, MAX_LENGTH);
        check_length(&g, &w, 4, 2, MAX_LENGTH);
        run_tests(&w, &vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        match g.precompute(initialize_state(optimizer::replace_seq(w).1)) {
            Fixed(_, _) => (), // Already checked the result above
            w => panic!(
                "Expected the Append to be precomputed to a Fixed, but got {:?}",
                w
            ),
        }
    }

    #[test]
    fn test_sum() {
        let g = new_test_generator(1);
        let w = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(1.0, 5, 2)),
            Box::new(finite_const_waveform(1.0, 5, 2)),
        );
        check_offset(&g, &w, 4);
        check_length(&g, &w, 0, 7, MAX_LENGTH);
        run_tests(&w, &vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]);

        let w2 = Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time(())),
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
        run_tests(&w2, &vec![3.0; 8]);

        let w5 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 2, 2)),
        );
        run_tests(&w5, &vec![3.0, 0.0, 0.0, 2.0, 2.0]);

        // Test a case to make sure that the sum generates enough samples, even when
        // the left-hand side is shorter and the right hasn't started yet.
        let (_, w5_no_seq) = optimizer::replace_seq(w5);
        let (_, result) = g.generate(initialize_state(w5_no_seq.clone()), 2);
        assert_eq!(result, vec![3.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 0, 0)),
        );
        let (_, w6_no_seq) = optimizer::replace_seq(w6);
        let (_, result) = g.generate(initialize_state(w6_no_seq), 2);
        assert_eq!(result, vec![3.0, 0.0]);

        let w7 = BinaryPointOp(
            Operator::Add,
            Box::new(Fixed(vec![1.0], ())),
            Box::new(Const(0.0)),
        );
        run_tests(&w7, &vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let g = new_test_generator(1);
        let w1 = BinaryPointOp(
            Operator::Multiply,
            Box::new(finite_const_waveform(3.0, 8, 2)),
            Box::new(finite_const_waveform(2.0, 5, 2)),
        );
        check_offset(&g, &w1, 4);
        check_length(&g, &w1, 0, 7, MAX_LENGTH);
        run_tests(&w1, &vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0]);

        let w2 = BinaryPointOp(
            Operator::Multiply,
            Box::new(finite_const_waveform(3.0, 5, 2)),
            Box::new(finite_const_waveform(2.0, 5, 2)),
        );
        run_tests(&w2, &vec![3.0, 3.0, 6.0, 6.0, 6.0]);

        let w3 = Fin {
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
        run_tests(&w3, &vec![6.0; 8]);

        let w4 = BinaryPointOp(
            Operator::Multiply,
            Box::new(Seq {
                offset: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(1.0)),
                )),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_waveform(2.0, 5, 5)),
        );
        run_tests(&w4, &vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0]);

        let w = BinaryPointOp(
            Operator::Multiply,
            Box::new(finite_const_waveform(3.0, 6, 2)),
            Box::new(Const(2.0)),
        );
        check_offset(&g, &w, 2);
        check_length(&g, &w, 0, 6, MAX_LENGTH);
        run_tests(&w, &vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0]);
        match g.precompute(initialize_state(optimizer::replace_seq(w).1)) {
            Fixed(_, _) => (), // Already checked the result above
            w => panic!(
                "Expected the Append to be precomputed to a Fixed, but got {:?}",
                w
            ),
        }
    }

    #[test]
    fn test_filter() {
        let g = new_test_generator(1);

        // FIRs
        let w1 = Filter {
            waveform: Box::new(Time(())),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        run_tests(&w1, &vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]);

        let w2 = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(5.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        run_tests(&w2, &vec![6.0, 12.0, 18.0]);

        let w = Filter {
            waveform: Box::new(Fixed(vec![1.0, 2.0, 3.0], ())),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0, 2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        check_length(&g, &w, 0, 0, 5);

        let w3 = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(8.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0, 2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        let g = new_test_generator(1);
        check_length(&g, &w3, 0, 4, MAX_LENGTH);
        run_tests(&w3, &vec![20.0, 30.0, 40.0, 50.0]);

        let w4 = Filter {
            waveform: Box::new(Res {
                // Pick a trigger that's far from zero on at our sampled points
                trigger: sin_waveform(1.0 / 3.0, 3.0 * std::f32::consts::PI / 2.0),
                waveform: Box::new(Time(())),
                state: (),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        run_tests(&w4, &vec![0.0, 2.0, 6.0, 4.0, 2.0, 6.0, 4.0, 2.0]);

        let w5 = Filter {
            waveform: Box::new(Const(1.0)),
            feed_forward: Box::new(Fixed(vec![0.2, 0.2, 0.2, 0.2, 0.2], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        run_tests(&w5, &vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // IIRs
        let w6 = Filter {
            waveform: Box::new(Time(())),
            feed_forward: Box::new(Fixed(vec![0.5], ())),
            feedback: Box::new(Fixed(vec![-0.5], ())),
            state: (),
        };
        run_tests(
            &w6,
            &vec![0.0, 0.5, 1.25, 2.125, 3.0625, 4.03125, 5.015625, 6.0078125],
        );

        // Cascade
        let w7 = Filter {
            waveform: Box::new(Filter {
                waveform: Box::new(Time(())),
                feed_forward: Box::new(Fixed(vec![0.5], ())),
                feedback: Box::new(Fixed(vec![-0.5], ())),
                state: (),
            }),
            feed_forward: Box::new(Fixed(vec![0.4], ())),
            feedback: Box::new(Fixed(vec![-0.6], ())),
            state: (),
        };
        run_tests(
            &w7,
            &vec![
                0.0, 0.2, 0.62, 1.222, 1.9582, 2.7874203, 3.6787024, 4.610347,
            ],
        );
    }

    #[test]
    fn test_greater_or_equals_at() {
        let w1 = BinaryPointOp(Operator::Add, Box::new(Time(())), Box::new(Const(-5.0)));
        let w2 = Fin {
            length: Box::new(w1.clone()),
            waveform: Box::new(Time(())),
        };
        let g = new_test_generator(1);
        let position = g.greater_or_equals_at(&initialize_state(w1.clone()), 0.0, 10);
        let (_, out) = g.generate(initialize_state(w2), 10);
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
