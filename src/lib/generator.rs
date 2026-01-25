use std::cell::RefCell;
use std::collections::HashMap;
use std::f64;
use std::fmt::Debug;
use std::io::BufWriter;

use fastrand;

use crate::waveform;

#[derive(Debug, Clone)]
pub enum State {
    // The position in the current waveform as the number of samples since the beginning of the waveform.
    Position {
        position: i64,
        direction: i64,
    },
    Remaining(usize),
    // TODO consider a shared vec plus a slice for Fixed? how to represent a negative position? and direction?
    // Shared(Rc<Vec<f32>>, &[f32]),
    // Previously generated samples as input and/or output to the waveform. (for example, for Filter)
    Samples {
        input: Vec<f32>,
        output: Vec<f32>,
    },
    // The current accumulated phase as well the previous value of the frequency and phase offset waveforms
    // (for example, for Sin).
    Phase {
        accumulator: f64,
        direction: f64,
        previous_frequency: f32,
        previous_phase_offset: f32,
    },
    // The sign of the last generated value along with the direction of generation (for example, for Res).
    Sign {
        signum: f32,
        direction: i64,
    },
    // For cases that don't require any state.
    Empty,
    // For cases that haven't yet been initialized.
    Uninitialized,
}

fn new_position(position: i64, direction: i64) -> State {
    return State::Position {
        position,
        direction,
    };
}

pub const INITIAL_STATE: State = State::Position {
    position: 0,
    direction: 1,
};

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
    // Note that the returned waveform should support being "re-initialized" -- unless it depends on some external
    // state (as Slider does), it should generate the same samples each time it's initialized with Position(0, 1).
    // Note that position can be negative and that in that case, this will always return at least
    // `desired.min(position.abs())` samples some of which may be 0.0 (that is, waveforms are not allowed to end
    // before position 0.0). Similarly, if direction is negative, this will always return `desired` samples (that is,
    // if `position` is greater than the length of the waveform and direction is negative, then the initial samples
    // will be 0.0 and any samples for negative positions may also be 0.0).
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
            Time(Position {
                position,
                direction,
            }) => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = (position + direction * i as i64) as f32 / self.sample_frequency as f32;
                }
                return (
                    Time(new_position(
                        position + direction * desired as i64,
                        direction,
                    )),
                    out,
                );
            }
            Time(_) => unreachable!("Time waveform has non-Position state"),
            Noise => {
                let mut out = vec![0.0; desired];
                for x in out.iter_mut() {
                    *x = fastrand::f32() * 2.0 - 1.0;
                }
                return (waveform, out);
            }
            Fixed(
                samples,
                Position {
                    position,
                    direction,
                },
            ) => {
                let mut desired = desired;
                if direction > 0 {
                    let mut out = vec![];
                    let tmp_position;
                    if position < 0 {
                        // Pad with zeros for the negative positions
                        out = vec![0.0; desired.min(-position as usize)];
                        tmp_position = 0;
                        desired -= out.len();
                    } else {
                        tmp_position = position as usize;
                    }
                    if tmp_position < samples.len() {
                        out.extend(
                            &samples[tmp_position..(tmp_position + desired).min(samples.len())],
                        );
                    }
                    return (
                        Fixed(
                            samples,
                            new_position(position + out.len() as i64, direction),
                        ),
                        out,
                    );
                } else {
                    // direction < 0
                    let mut out = vec![];
                    let (start, end); // upper bound to the copy below
                    if position > samples.len() as i64 {
                        // Pad with zeros beyond samples.len()
                        out = vec![0.0; desired.min((position - samples.len() as i64) as usize)];
                        end = samples.len();
                        start = end - (desired - out.len()).min(end);
                    } else if position > 0 {
                        end = position as usize;
                        start = end - desired.min(end);
                    } else {
                        (start, end) = (0, 0); // not exactly bounds, but the right thing will happen below
                    }
                    out.extend(&mut samples[start..end].iter().rev());
                    // Pad with zeros for negative positions (which should be all that's left).
                    out.extend(&vec![0.0; desired - out.len()]);
                    return (
                        Fixed(
                            samples,
                            new_position(position - out.len() as i64, direction),
                        ),
                        out,
                    );
                }
            }
            Fixed(_, _) => unreachable!("Fixed waveform has non-Position state"),
            Fin {
                length,
                waveform: inner,
            } => {
                // XXXXX negative direction?
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
            Append(
                a,
                b,
                state @ Position {
                    position,
                    direction,
                },
            ) => {
                // Note that any negative positions always come from `a`, regardless of the `direction`.
                if direction > 0 {
                    // In this case, we've yet to generate any samples and `position` is the initial position.
                    let a = self.initialize_state(*a, state);
                    let (a, mut a_out) = self.generate(a, desired);
                    if a_out.len() == desired {
                        // We know that the initial position was accounted for by `a`, so we can initialize `b`
                        // with a position of 0.
                        let b = self.initialize_state(*b, new_position(0, direction));
                        // Keep what's left of `a` in the waveform.
                        return (Append(Box::new(a), Box::new(b), Empty), a_out);
                    } else {
                        // a_out.len() < desired
                        let mut b = *b;
                        if a_out.is_empty() && position > 0 {
                            // Find out if `a` was long enough to get us to `position` or not.
                            let (_, total_a_len) = self.remaining(
                                self.initialize_state(a.clone(), new_position(0, direction)),
                                position as usize,
                            );
                            // We need to generate some from `b` right now, so initialize it with what's left of the
                            // initial position.
                            b = self.initialize_state(
                                b,
                                new_position(position - total_a_len as i64, direction),
                            );
                        } else {
                            b = self.initialize_state(b, new_position(0, direction));
                        }
                        // Keep `a` around in case this result is ever reset. Switch the state to an Empty state to
                        // indicate that we've properly accounted for the initial position w.r.t. `b`.
                        let (waveform, mut b_out) = self.generate(
                            Append(Box::new(a), Box::new(b), Empty),
                            desired - a_out.len(),
                        );
                        a_out.append(&mut b_out);
                        return (waveform, a_out);
                    }
                } else {
                    // direction < 0
                    // If direction is negative, we need to see if position is greater than the length of `a`
                    // and by how much.
                    let (_, total_a_len) = self.remaining(
                        self.initialize_state(*a.clone(), new_position(0, 1)),
                        if position >= 0 {
                            position as usize + 1
                        } else {
                            0
                        },
                    );
                    if total_a_len as i64 > position {
                        // `b` won't be used: just generate from `a`
                        let a = self.initialize_state(*a, state);
                        let (a, out) = self.generate(a, desired);
                        return (Append(Box::new(a), b, Remaining(0)), out);
                    } else {
                        // total_a_len <= position, so we need part of `b`
                        let b = self.initialize_state(
                            *b,
                            new_position(position - total_a_len as i64, direction),
                        );
                        let a =
                            self.initialize_state(*a, new_position(total_a_len as i64, direction));
                        return self.generate(
                            Append(
                                Box::new(a),
                                Box::new(b),
                                Remaining(position as usize - total_a_len),
                            ),
                            desired,
                        );
                    }
                }
            }
            Append(a, b, Empty) => {
                // This is the positive direction case.
                // `a` may or may not have more samples, but we know that `b` is initialized.
                let (a, mut out) = self.generate(*a, desired);
                if out.len() == desired {
                    return (Append(Box::new(a), b, Empty), out);
                } else {
                    let (b, mut b_out) = self.generate(*b, desired - out.len());
                    out.append(&mut b_out);
                    return (Append(Box::new(a), Box::new(b), Empty), out);
                }
            }
            Append(a, b, Remaining(remaining)) => {
                // This is the negative direction case. `remaining` tells us the maximum number of samples that
                // we'll get from `b` (because after that position will be negative).
                let (b, mut out) = self.generate(*b, desired.min(remaining));
                if out.len() == desired {
                    return (
                        Append(a, Box::new(b), Remaining(remaining - out.len())),
                        out,
                    );
                } else {
                    let (a, mut a_out) = self.generate(*a, desired - out.len());
                    out.append(&mut a_out);
                    return (Append(Box::new(a), Box::new(b), Remaining(0)), out);
                }
            }
            Append(_, _, _) => unreachable!("Append waveform has non-Position, non-Samples state"),
            Sin {
                frequency,
                phase,
                state:
                    Position {
                        position,
                        direction,
                    },
            } => {
                // First, we need to compute the value of the phase accumulator at `position`.
                // Frequency is going to always be one position "ahead" of `waveform` so we generate one
                // value at the origin now.
                let tmp_frequency =
                    self.initialize_state(*frequency.clone(), new_position(0, position.signum()));
                let (tmp_frequency, f_out) = self.generate(tmp_frequency, 1);
                // For the phase offset, we want to know the "earlier" offset.
                let tmp_phase = self.initialize_state(
                    *phase.clone(),
                    new_position(-position.signum(), position.signum()),
                );
                let (tmp_phase, ph_out) = self.generate(tmp_phase, 1);
                if f_out.is_empty() || ph_out.is_empty() {
                    return (
                        Sin {
                            frequency: Box::new(*frequency),
                            phase: Box::new(*phase),
                            state: new_position(position, direction),
                        },
                        vec![],
                    );
                }
                // Now actually generate from 0 up to (or down to) `position` so that we can accumulate the
                // correct state.
                let (mut waveform, _) = self.generate(
                    Sin {
                        frequency: Box::new(tmp_frequency),
                        phase: Box::new(tmp_phase),
                        state: Phase {
                            accumulator: ph_out[0] as f64,
                            direction: position.signum() as f64,
                            previous_frequency: f_out[0],
                            previous_phase_offset: ph_out[0],
                        },
                    },
                    position.abs() as usize,
                );
                // Second, if the direction of the result is different than what we used to set the phase accumulator,
                // then reset and regenerate from the `frequency`` and `phase` waveforms, this time from `position`
                // and in the desired `direction``.
                if direction != position.signum() {
                    let frequency =
                        self.initialize_state(*frequency, new_position(position, direction));
                    let (frequency, f_out) = self.generate(frequency, 1);
                    // Phase offset still starts one position "earlier".
                    let phase = self
                        .initialize_state(*phase, new_position(position - direction, direction));
                    let (phase, ph_out) = self.generate(phase, 1);
                    waveform = if let Sin {
                        state: Phase { accumulator, .. },
                        ..
                    } = waveform
                    {
                        Sin {
                            frequency: Box::new(frequency),
                            phase: Box::new(phase),
                            state: Phase {
                                accumulator,
                                direction: direction as f64,
                                previous_frequency: f_out[0],
                                previous_phase_offset: ph_out[0],
                            },
                        }
                    } else {
                        unreachable!("generate(Sin) returned a non-Sin waveform");
                    }
                }
                // Discard any output generated this point and now generate the actual desired output.
                return self.generate(waveform, desired);
            }
            Sin {
                frequency,
                phase,
                state:
                    Phase {
                        mut accumulator,
                        direction,
                        mut previous_frequency,
                        mut previous_phase_offset,
                    },
            } => {
                // Instantaneous frequency (note that this is one position ahead of `waveform`).
                let (frequency, f_out) = self.generate(*frequency, desired);
                // Instantaneous phase offset.
                let (phase, ph_out) = self.generate(*phase, f_out.len());
                let mut out = vec![0.0; ph_out.len()];
                for (i, &phase_offset) in ph_out.iter().enumerate() {
                    out[i] = accumulator.sin() as f32;
                    let f = (previous_frequency + f_out[i]) / 2.0;
                    let phase_inc = f as f64 / self.sample_frequency as f64
                        + (phase_offset as f64 - previous_phase_offset as f64);
                    // Move the accumulator in the direction according to the frequency and
                    // change in phase offset.
                    accumulator =
                        (accumulator + direction * phase_inc).rem_euclid(f64::consts::TAU);
                    previous_frequency = f_out[i];
                    previous_phase_offset = phase_offset;
                }
                return (
                    Sin {
                        frequency: Box::new(frequency),
                        phase: Box::new(phase),
                        state: Phase {
                            accumulator,
                            direction,
                            previous_frequency,
                            previous_phase_offset,
                        },
                    },
                    out,
                );
            }
            Sin { .. } => unreachable!("Sin waveform has non-Position, non-Phase state"),
            Filter {
                waveform: inner,
                feed_forward,
                feedback,
                state:
                    Position {
                        position,
                        direction,
                    },
            } => {
                // XXX Double check `direction` for Filter... and what it should mean
                let (_, ff_len) = self.remaining(
                    self.initialize_state(*feed_forward.clone(), INITIAL_STATE),
                    MAX_FILTER_LENGTH,
                );
                // Generate the input samples from before `position` so that they can be used in the feed forward
                // part of the filter. Note that we could do this later (we could leave the inner waveform initialized
                // to the position below without generating here) but that complicates the management of the inner
                // waveform.
                let inner = self.initialize_state(
                    *inner,
                    Position {
                        position: position - (ff_len as i64 - 1),
                        direction,
                    },
                );
                let (inner, input) = self.generate(inner, ff_len - 1);
                let (_, fb_len) = self.remaining(
                    self.initialize_state(*feedback.clone(), INITIAL_STATE),
                    MAX_FILTER_LENGTH,
                );
                // Fill the previous output samples with zeros to match the length of `feedback`.
                let output = vec![0.0; fb_len];

                println!(
                    "Started generate for Filter({:?}, {:?}, {:?}) with {} samples for feed-forward and {} samples for feedback",
                    inner, feed_forward, feedback, ff_len, fb_len
                );

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
                let (_, ff_out) = self.generate(
                    self.initialize_state(*feed_forward.clone(), INITIAL_STATE),
                    MAX_FILTER_LENGTH,
                );
                let (_, fb_out) = self.generate(
                    self.initialize_state(*feedback.clone(), INITIAL_STATE),
                    MAX_FILTER_LENGTH,
                );

                // Set up the input and output. Each will have extra samples at the beginning from a previous
                // call to `generate`.
                let mut input = truncate_or_pad_left(previous_in, ff_out.len() - 1, 0.0);
                let (inner, mut inner_out) = self.generate(*inner, desired);
                input.append(&mut inner_out);
                assert!(input.len() >= ff_out.len() - 1); // XXXXX

                // Use `previous_out` to fill in samples in `out` that we'll use to compute the feedback part of the
                // filter. We'll trim off these extra samples at the end.
                let mut out = truncate_or_pad_left(previous_out, fb_out.len(), 0.0);
                assert_eq!(out.len(), fb_out.len()); // XXXXX
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
            Filter { .. } => unreachable!("Filter waveform has non-Position, non-Samples state"),
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
                waveform: inner,
                state:
                    state @ Position {
                        position,
                        direction,
                    },
            } => {
                // XXX Ugh... this constant...
                // Maximum number of samples to look back for a reset trigger
                let max_reset_lookback: i64 = self.sample_frequency as i64 * 10000;
                // TODO generate the trigger in blocks?

                // Go back and find the most recent trigger before `position`.
                let mut last_trigger_position = position;
                let tmp_trigger = self.initialize_state(
                    *trigger.clone(),
                    State::Position {
                        position: last_trigger_position,
                        direction: -direction,
                    },
                );
                let (mut tmp_trigger, t_out) = self.generate(tmp_trigger, 1);
                if t_out.is_empty() {
                    return (
                        Res {
                            trigger,
                            waveform: inner,
                            state,
                        },
                        vec![],
                    );
                }
                let mut last_signum = t_out[0].signum();
                let first_signum = last_signum;
                loop {
                    if position - last_trigger_position > max_reset_lookback {
                        panic!(
                            "No reset trigger found within {} samples before position {} in waveform {:?}",
                            max_reset_lookback, position, trigger
                        );
                    }
                    let (new_tmp, t_out) = self.generate(tmp_trigger, 1);
                    tmp_trigger = new_tmp;
                    let new_signum = t_out.get(0).unwrap().signum();
                    if last_signum >= 0.0 && new_signum < 0.0 {
                        break;
                    }
                    last_signum = new_signum;
                    last_trigger_position -= direction;
                }

                // Now initialize the inner waveform according to that position.
                let inner = self.initialize_state(
                    *inner,
                    Position {
                        position: position - last_trigger_position,
                        direction,
                    },
                );
                return self.generate(
                    Res {
                        trigger,
                        waveform: Box::new(inner),
                        state: Sign {
                            signum: first_signum,
                            direction,
                        },
                    },
                    desired,
                );
            }
            Res {
                trigger,
                waveform: inner,
                state:
                    Sign {
                        mut signum,
                        direction,
                    },
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
                        inner = self.initialize_state(
                            inner,
                            Position {
                                position: 0,
                                direction,
                            },
                        );
                    }
                    generated += inner_desired;
                }
                return (
                    Res {
                        trigger: Box::new(trigger),
                        waveform: Box::new(inner),
                        state: Sign { signum, direction },
                    },
                    out,
                );
            }
            Res { .. } => unreachable!("Res waveform has non-Position, non-Sign state"),
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
                //  - that means Sin wouldn't set previous_phases at position 0
                let (inner, out) = self.generate(*inner, desired);
                if self.capture_state.is_none() {
                    // This occurs, for example, in cases where we need to advance the position of a waveform
                    // XXX do we still need this now that remaining advances?
                    return (inner, out);
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
        // XXX implement direction.. starting with Fin
        match waveform {
            Const { .. } => (waveform, max),
            Time(Position {
                position,
                direction,
            }) => (
                Time(new_position(position + direction * max as i64, direction)),
                max,
            ),
            Time(_) => unreachable!("Time waveform with non-Position state"),
            Noise => (Noise, max),
            Fixed(
                samples,
                Position {
                    position,
                    direction,
                },
            ) => {
                if direction > 0 {
                    let mut len = 0;
                    let tmp_position;
                    if position < 0 {
                        len = max.min(-position as usize);
                        tmp_position = 0;
                    } else {
                        tmp_position = position as usize;
                    }
                    if tmp_position < samples.len() {
                        len += (max - len).min(samples.len());
                    }
                    return (
                        Fixed(samples, new_position(position + len as i64, direction)),
                        len,
                    );
                } else {
                    // Always return `max` in the negative direction
                    return (
                        Fixed(samples, new_position(position - max as i64, direction)),
                        max,
                    );
                }
            }
            Fixed(_, _) => unreachable!("Fixed waveform with non-Position state"),
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
                waveform: inner, ..
            }
            | Filter {
                waveform: inner, ..
            } => self.remaining(*inner, max),
            Append(
                a,
                b,
                state @ Position {
                    position,
                    direction,
                },
            ) => {
                if direction > 0 {
                    let a = self.initialize_state(*a, state);
                    let (a, a_len) = self.remaining(a, max);
                    if a_len == max {
                        let b = self.initialize_state(*b, new_position(0, direction));
                        return (Append(Box::new(a), Box::new(b), Empty), a_len);
                    } else {
                        // a_len < max
                        let mut b = *b;
                        if a_len == 0 && position > 0 {
                            let (_, total_a_len) = self.remaining(
                                self.initialize_state(a.clone(), new_position(0, direction)),
                                position as usize,
                            );
                            b = self.initialize_state(
                                b,
                                new_position(position - total_a_len as i64, direction),
                            );
                        } else {
                            b = self.initialize_state(b, new_position(0, direction));
                        }
                        let (waveform, b_len) =
                            self.remaining(Append(Box::new(a), Box::new(b), Empty), max - a_len);
                        return (waveform, a_len + b_len);
                    }
                } else {
                    // direction < 0
                    let (_, total_a_len) = self.remaining(
                        self.initialize_state(*a.clone(), new_position(0, 1)),
                        if position >= 0 {
                            position as usize + 1
                        } else {
                            0
                        },
                    );
                    if total_a_len as i64 > position {
                        let a = self.initialize_state(*a, state);
                        let (a, len) = self.remaining(a, max);
                        return (Append(Box::new(a), b, Remaining(0)), len);
                    } else {
                        // total_a_len <= position, so we need part of `b`
                        let b = self.initialize_state(
                            *b,
                            new_position(position - total_a_len as i64, direction),
                        );
                        let a =
                            self.initialize_state(*a, new_position(total_a_len as i64, direction));
                        return self.remaining(
                            Append(
                                Box::new(a),
                                Box::new(b),
                                Remaining(position as usize - total_a_len),
                            ),
                            max,
                        );
                    }
                }
            }
            Append(a, b, Empty) => {
                let (a, a_len) = self.remaining(*a, max);
                let (b, b_len) = self.remaining(*b, max - a_len);
                (Append(Box::new(a), Box::new(b), Empty), a_len + b_len)
            }
            Append(a, b, Remaining(remaining)) => {
                let (b, b_len) = self.remaining(*b, max.min(remaining));
                let (a, a_len) = self.remaining(*a, max - b_len);
                (
                    Append(Box::new(a), Box::new(b), Remaining(remaining - b_len)),
                    a_len + b_len,
                )
            }
            Append(_, _, _) => {
                unreachable!("Append with non-Position, non-Empty, non-Remaining state")
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
        use State::{Position, Samples};
        use waveform::Operator;
        use waveform::Waveform::{Append, BinaryPointOp, Const, Time};
        match waveform {
            Const(v) if *v >= value => MaybeOption::Some(0),
            Const(_) => MaybeOption::None,
            &Time(Position {
                position,
                direction,
            }) => {
                let current_value = position as f32 / self.sample_frequency as f32;
                if direction > 0 {
                    if current_value >= value {
                        MaybeOption::Some(0)
                    } else {
                        let target_position = (value * self.sample_frequency as f32).ceil() as i64;
                        MaybeOption::Some(max.min((target_position - position) as usize))
                    }
                } else {
                    // direction < 0
                    if current_value >= value {
                        MaybeOption::Some(0)
                    } else {
                        MaybeOption::None
                    }
                }
            }
            Append(a, b, state) => {
                match self.greater_or_equals_at(a, value, max) {
                    MaybeOption::None => {
                        let (a, a_len) = self.remaining(*a.clone(), max);
                        if a_len == max {
                            // We didn't reach the end of `a``, so `b` isn't relevant yet.
                            MaybeOption::None
                        } else if let Position {
                            position,
                            direction,
                        } = state
                        {
                            // XXX impl direction
                            let b = self.initialize_state(
                                *b.clone(),
                                Position {
                                    position: (*position - a_len as i64).max(0),
                                    direction: *direction,
                                },
                            );
                            self.greater_or_equals_at(
                                &Append(
                                    Box::new(a),
                                    Box::new(b),
                                    Samples {
                                        input: vec![],
                                        output: vec![],
                                    },
                                ),
                                value,
                                max,
                            )
                        } else if let Samples { .. } = state {
                            // `b` is already initialized
                            match self.greater_or_equals_at(&b, value, max - a_len) {
                                MaybeOption::Some(size) => MaybeOption::Some(size + a_len),
                                m => m,
                            }
                        } else {
                            unreachable!("Append has non-Position, non-Samples state")
                        }
                    }
                    m => m, // Some and Maybe get passed through
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
            Append(a, b, _) => (self.offset(a, max) + self.offset(b, max)).min(max),
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

    pub fn initialize_state<S>(&self, waveform: waveform::Waveform<S>, state: State) -> Waveform {
        use waveform::Waveform::*;
        match waveform {
            Const(value) => Const(value),
            Time(_) => Time(state),
            Noise => Noise,
            Fixed(samples, _) => Fixed(samples, state),
            Fin { length, waveform } => Fin {
                length: Box::new(self.initialize_state(*length, state.clone())),
                waveform: Box::new(self.initialize_state(*waveform, state)),
            },
            Seq { .. } => unreachable!("Can't initialize state for Seq"),
            /* Seq {
                offset: Box::new(self.initialize_state(*offset, state.clone())),
                waveform: Box::new(self.initialize_state(*waveform, state)),
            }, */
            Append(a, b, _) => Append(
                Box::new(self.initialize_state(*a, State::Uninitialized)),
                Box::new(self.initialize_state(*b, State::Uninitialized)),
                state,
            ),
            Sin {
                frequency, phase, ..
            } => Sin {
                frequency: Box::new(self.initialize_state(*frequency, State::Uninitialized)),
                phase: Box::new(self.initialize_state(*phase, State::Uninitialized)),
                state: state,
            },
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => Filter {
                // Don't initialize the inner waveform until we can determine the length of feed_forward
                waveform: Box::new(self.initialize_state(*waveform, state.clone())),
                feed_forward: Box::new(self.initialize_state(*feed_forward, State::Uninitialized)),
                feedback: Box::new(self.initialize_state(*feedback, State::Uninitialized)),
                state: state,
            },
            BinaryPointOp(op, a, b) => BinaryPointOp(
                op,
                Box::new(self.initialize_state(*a, state.clone())),
                Box::new(self.initialize_state(*b, state)),
            ),
            Res {
                trigger, waveform, ..
            } => Res {
                trigger: Box::new(self.initialize_state(*trigger, state.clone())),
                waveform: Box::new(self.initialize_state(*waveform, State::Uninitialized)),
                state: state,
            },
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => Alt {
                trigger: Box::new(self.initialize_state(*trigger, state.clone())),
                positive_waveform: Box::new(
                    self.initialize_state(*positive_waveform, state.clone()),
                ),
                negative_waveform: Box::new(self.initialize_state(*negative_waveform, state)),
            },
            Slider(slider) => Slider(slider),
            Marked { id, waveform } => Marked {
                id,
                waveform: Box::new(self.initialize_state(*waveform, state)),
            },
            Captured {
                file_stem,
                waveform,
            } => Captured {
                file_stem,
                waveform: Box::new(self.initialize_state(*waveform, state)),
            },
        }
    }
}

/*
// Replaces parts of `waveform` that can be precomputed with their precomputed Fixed versions. Notably,
// infinite waveforms and waveforms that depend on or have dynamic behavior (Slider, Marked, Captured)
// cannot be precomputed. This should be called after remove_seq.
//
// N.B. This isn't currently safe as values at negative positions aren't considered; for example, time is
// negative before zero, but a fixed waveform is always zero before its start.
pub fn precompute(g: &Generator, waveform: Waveform) -> Waveform {
    enum Result {
        Precomputed(Waveform), // This is always a Fixed waveform
        Infinite(Waveform),
        Dynamic(Waveform),
    }

    impl Into<Waveform> for Result {
        fn into(self) -> Waveform {
            match self {
                Result::Precomputed(w) => w,
                Result::Infinite(w) => w,
                Result::Dynamic(w) => w,
            }
        }
    }

    fn precompute_internal(g: &Generator, waveform: Waveform) -> Result {
        use Result::*;
        use waveform::Operator;
        use waveform::Waveform::*;

        // TODO how do we feel about all of the usize::MAX here?
        // TODO there's slightly more repetition than I'd like... for example, lots of "if the child is Dynamic,
        // then parent is Dynamic"

        fn generate_fixed(
            g: &Generator,
            waveform: Waveform,
        ) -> Result {
            let (_, out) = g.generate(waveform, usize::MAX);
            Precomputed(Fixed(
                out,
                State::Position(0), // we've already accounted for the position in waveform
            ))
        }

        match waveform {
            Const(_) | Time(_) | Noise => Infinite(waveform),
            Fixed(_, _) => Precomputed(waveform),
            Fin { length, waveform } => match precompute_internal(g, *waveform) {
                Precomputed(waveform) => {
                    generate_fixed(g, Fin { length, waveform: Box::new(waveform) })
                },
                Infinite(waveform) => {
                    generate_fixed(g, Fin { length, waveform: Box::new(waveform) })
                }
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
            Append(a, b, state) => match (
                precompute_internal(g, *a),
                precompute_internal(g, *b),
            ) {
                (Precomputed(Fixed(a,position_a)), Precomputed(Fixed(b, position_b))) => {
                    // XXXXX positions?
                    let v = [a, b].concat();
                    Precomputed(Fixed(v, State::Position(0))) // XXXXX position
                }
                (Precomputed(a), Precomputed(b)) => unreachable!("Precomputed waveforms should always be Fixed"),
                (Infinite(a), Infinite(b)) |
                (Infinite(a), Precomputed(b)) |
                (Precomputed(a), Infinite(b)) => {
                    Infinite(Append(Box::new(a), Box::new(b)))
                }
                // At least one is Dynamic
                (a, b) => Dynamic(Append(Box::new(a.into()), Box::new(b.into()))),
            },
            Sin {
                frequency,
                phase,
                state,
            } => match (
                precompute_internal(g, *frequency),
                precompute_internal(g, *phase),
            ) {
                (Precomputed(f), Precomputed(p)) => generate_fixed(
                    g,
                    Sin {
                        frequency: Box::new(f),
                        phase: Box::new(p),
                        state,
                    },
                ),
                (Infinite(f), Infinite(p)) |
                (Infinite(f), Precomputed(p)) |
                (Precomputed(f), Infinite(p)) => {
                    Infinite(Sin { frequency: Box::new(f), phase: Box::new(p), state })
                },
                // At least one is Dynamic
                (f, p) => Dynamic(Sin {
                    frequency: Box::new(f.into()),
                    phase: Box::new(p.into()),
                    state,
                }),
            },
            BinaryPointOp(op, a, b) => match (
                precompute_internal(g, *a),
                precompute_internal(g, *b),
            ) {
                (Precomputed(a), Precomputed(b)) => Precomputed(g.generate(
                    &BinaryPointOp(op, Box::new(Fixed(a)), Box::new(Fixed(b))),
                    0,
                    usize::MAX,
                )),
                (Infinite(a), Infinite(b)) => Infinite(BinaryPointOp(op, Box::new(a), Box::new(b))),
                (Infinite(a), Precomputed(b)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(a), Box::new(b)))
                    }
                    Operator::Multiply | Operator::Divide => generated_fixed(
                        g,
                        BinaryPointOp(op, Box::new(a), Box::new(b)),
                    ),
                },
                (Precomputed(v), Infinite(b)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(Fixed(v)), Box::new(b)))
                    }
                    Operator::Multiply | Operator::Divide => Precomputed(g.generate(
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
                    precompute_internal(g, *waveform),
                    precompute_internal(g, *feed_forward),
                    precompute_internal(g, *feedback),
                ) {
                    (Precomputed(w), Precomputed(ff), Precomputed(fb)) => {
                        let ff_len = ff.len();
                        Precomputed(g.generate(
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
                    precompute_internal(g, *trigger),
                    precompute_internal(g, *waveform),
                ) {
                    (Precomputed(t), Precomputed(w)) => Precomputed(g.generate(
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
                    precompute_internal(g, *trigger),
                    precompute_internal(g, *positive_waveform),
                    precompute_internal(g, *negative_waveform),
                ) {
                    (Precomputed(t), Precomputed(p), Precomputed(n)) => {
                        Precomputed(g.generate(
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

    return precompute_internal(g, waveform).into();
}
*/

#[cfg(test)]
mod tests {
    use core::f64;
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
        let offset_waveform = g.initialize_state(offset_waveform, INITIAL_STATE);
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
    fn check_length(g: &Generator, waveform: &Waveform, state: State, expected: usize, max: usize) {
        let (_, waveform) = optimizer::replace_seq(waveform.clone());
        let waveform = g.initialize_state(waveform, state);
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
        check_length(&g, &waveform, INITIAL_STATE, desired.len(), desired.len());
        /*
        // We can't currently generate for waveforms that contain Seq, so skip this.
        for size in [1, 2, 4, 8] {
            let g = new_test_generator(1);
            let mut w = g.initialize_state(waveform.clone(), INITIAL_STATE);
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
            check_length(
                &g,
                &no_seq_waveform,
                INITIAL_STATE,
                desired.len(),
                desired.len(),
            );
            let mut w = g.initialize_state(no_seq_waveform.clone(), INITIAL_STATE);
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
            check_length(
                &g,
                &optimized_waveform,
                INITIAL_STATE,
                desired.len(),
                desired.len(),
            );
            let mut w = g.initialize_state(optimized_waveform.clone(), INITIAL_STATE);
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

        /*
        for size in [1, 2, 4, 8] {
            let g = new_test_generator(1);
            let (_, no_seq_waveform) = optimizer::replace_seq(waveform.clone());
            let optimized_waveform = optimizer::simplify(no_seq_waveform.clone(), INITIAL_STATE);
            let mut w = optimizer::precompute(&generator, initialize_state(optimized_waveform.clone()));
            check_length(&g, &w, INITIAL_STATE, desired.len(), desired.len());
            let mut out = vec![0.0; desired.len()];
            for n in 0..out.len() / size {
                let (w, tmp) = g.generate(w, size);
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
                size, waveform, remove_state(w)
            );
        }
        */
    }

    #[test]
    fn test_time() {
        let w = Time(());
        run_tests(&w, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let g = new_test_generator(1);
        let (_, result) = g.generate(g.initialize_state(w, new_position(4, -1)), 8);
        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_fixed() {
        let w = Fixed(vec![1.0, 2.0, 3.0, 4.0, 5.0], ());
        run_tests(&w, &vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let g = new_test_generator(1);
        let (_, result) = g.generate(g.initialize_state(w.clone(), new_position(6, 1)), 8);
        assert_eq!(result, vec![]);

        let (_, result) = g.generate(g.initialize_state(w.clone(), new_position(-2, 1)), 8);
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        let (_, result) = g.generate(g.initialize_state(w.clone(), new_position(6, -1)), 8);
        assert_eq!(result, vec![0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]);

        let (_, result) = g.generate(g.initialize_state(w, new_position(8, -1)), 2);
        assert_eq!(result, vec![0.0, 0.0]);
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
        let w = g.initialize_state(*sin_waveform(1.0, 0.0), INITIAL_STATE);
        let expected = (0..100)
            .map(|x: i32| (f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&g, &w, expected);

        // Start at non-zero position
        let w = g.initialize_state(*sin_waveform(1.0, 0.0), new_position(10, 1));
        let expected = (10..100)
            .map(|x: i32| (f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&g, &w, expected);

        // Go backward
        let w = g.initialize_state(*sin_waveform(1.0, 0.0), new_position(0, -1));
        let expected = (0..100)
            .map(|x: i32| (-f64::consts::TAU * x as f64 / sample_frequency).sin() as f32)
            .collect();
        run_sin_test(&g, &w, expected);

        // Non-constant frequency: f = time + 10 Hz
        let w3 = g.initialize_state(
            Waveform::Sin {
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
            },
            INITIAL_STATE,
        );
        let f_is_t_plus_ten = |x: i32| {
            let t = x as f64 / sample_frequency;
            let phase = f64::consts::TAU * (0.5 * t * t + 10.0 * t);
            phase.sin() as f32
        };
        //let expected = (0..100).map(f_is_t_plus_ten).collect();
        //run_sin_test(&g, &w3, expected);

        // Negative position, but going forward
        let w = g.initialize_state(w3, new_position(-10, 1));
        let expected = (-10..90).map(f_is_t_plus_ten).collect();
        run_sin_test(&g, &w, expected);

        // Non-zero phase offset, going backward
        let w = g.initialize_state(
            *sin_waveform(0.25, 5.0 * f32::consts::PI / 4.0),
            new_position(0, -1),
        );
        let expected: Vec<_> = (0..100)
            .map(|x: i32| {
                (f64::consts::TAU * 0.25 * (-x as f64 / sample_frequency)
                    + 5.0 * f64::consts::PI / 4.0)
                    .sin() as f32
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

        // Test a reset that occurs before time 0
        let w4 = Res {
            trigger: Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Time(())),
                Box::new(Const(2.0)),
            )),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w4, &vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let w5 = Res {
            trigger: sin_waveform(0.25, f32::consts::PI),
            waveform: Box::new(Time(())),
            state: (),
        };
        run_tests(&w5, &vec![2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]);

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
            (),
        );

        check_offset(&g, &w, 4);
        check_length(&g, &w, INITIAL_STATE, 6, MAX_LENGTH);
        check_length(&g, &w, new_position(2, 1), 4, MAX_LENGTH);
        check_length(&g, &w, new_position(4, 1), 2, MAX_LENGTH);
        run_tests(&w, &vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        /*
        // Negative direction XXX
        check_length(&g, &w, new_position(2, -1), 3, 3);
        check_length(&g, &w, new_position(4, -1), 5, 5);
        let (_, out) = g.generate(g.initialize_state(w, new_position(4, -1)), MAX_LENGTH);
        assert_eq!(out, vec![2.0, 1.0, 1.0, 1.0]);
        */
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
        check_length(&g, &w, INITIAL_STATE, 7, MAX_LENGTH);
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
        let (_, result) = g.generate(g.initialize_state(w5_no_seq.clone(), INITIAL_STATE), 2);
        assert_eq!(result, vec![3.0, 0.0]);
        let (_, result) = g.generate(g.initialize_state(w5_no_seq.clone(), new_position(1, 1)), 2);
        assert_eq!(result, vec![0.0, 0.0]);

        // This one is a little strange: the right-hand side doesn't generate any
        // samples but we still want length(a ~+ b) to be
        //   max(length(a), offset(a) + length(b)).
        let w6 = BinaryPointOp(
            Operator::Add,
            Box::new(finite_const_waveform(3.0, 1, 3)),
            Box::new(finite_const_waveform(2.0, 0, 0)),
        );
        let (_, w6_no_seq) = optimizer::replace_seq(w6);
        let (_, result) = g.generate(g.initialize_state(w6_no_seq, INITIAL_STATE), 2);
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
        check_length(&g, &w1, INITIAL_STATE, 7, MAX_LENGTH);
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
        run_tests(&w1, &vec![-6.0, 0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0]);

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
        check_length(&g, &w2, INITIAL_STATE, 5, MAX_LENGTH);
        run_tests(&w2, &vec![-6.0, 0.0, 6.0, 12.0, 18.0]);

        let w3 = Filter {
            waveform: Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Time(())),
                    Box::new(Const(3.0)),
                )),
                waveform: Box::new(Time(())),
            }),
            feed_forward: Box::new(Fixed(vec![2.0, 2.0, 2.0, 2.0, 2.0], ())),
            feedback: Box::new(Fixed(vec![], ())),
            state: (),
        };
        let g = new_test_generator(1);
        check_length(&g, &w3, INITIAL_STATE, 3, MAX_LENGTH);
        run_tests(&w3, &vec![-20.0, -10.0, 0.0]);

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
        run_tests(&w4, &vec![6.0, 4.0, 2.0, 6.0, 4.0, 2.0, 6.0, 4.0]);

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
        let position =
            g.greater_or_equals_at(&g.initialize_state(w1.clone(), INITIAL_STATE), 0.0, 10);
        let (_, out) = g.generate(g.initialize_state(w2, INITIAL_STATE), 10);
        assert!(position.is_some());
        assert_eq!(position.unwrap(), out.len());
        for (i, x) in out.iter().enumerate() {
            if i < position.unwrap() {
                assert_eq!(*x, i as f32);
            } else if i == position.unwrap() {
                assert!(*x >= 0.0);
            }
        }

        let position =
            g.greater_or_equals_at(&g.initialize_state(w1, new_position(10, -1)), 0.0, 10);
        assert!(position.is_some());
        assert_eq!(position.unwrap(), 0);
    }

    // TODO test for forgetting to sort pending waveforms
}
