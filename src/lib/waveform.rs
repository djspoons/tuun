use std::fmt;
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    /// Computes the sum of two samples.
    Add,
    /// Computes the difference of two samples.
    Subtract,
    /// Computes the product of two samples.
    Multiply,
    /// Computes the ratio of two samples or yields zero if the second sample is zero.
    Divide,
    /// Computes the sum of two samples or, in the case where only one sample is defined,
    /// returns that sample.
    Merge,
}

/// Waveform is a compact representation of a sequence of samples.
#[derive(Debug, Clone, PartialEq)]
pub enum Waveform<MarkId, State = ()> {
    /// Produces a stream of samples where each sample is the same constant value.
    Const(f32),
    /// Generates a stream based on the elapsed time from the start of the waveform, in seconds.
    Time(State),
    /// Generates random samples.
    Noise,
    /// Generates a finite sequence of samples.
    Fixed(Vec<f32>, State),
    /// Generates a finite waveform, truncating the underlying waveform. The length is determined
    /// by the first point at which the `length` waveform is >= 0.0. For example, `Fin(Const(0.0), _)`
    /// is 0 seconds in length and `Fin(Subtract(Time, Const(2.0)), _)` is 2 seconds in length.
    Fin {
        length: Box<Waveform<MarkId, State>>,
        waveform: Box<Waveform<MarkId, State>>,
    },
    /// Concatenates two waveforms, generating all samples from the first waveform and
    /// then all samples from the second.
    Append(Box<Waveform<MarkId, State>>, Box<Waveform<MarkId, State>>),
    /// Computes the sine with the given frequency and phase (both in radians). Equivalently, computes
    /// the sine of angle that changes according to the rate of the first parameter and the value of the
    /// second. Note that Sine is used both as the basis for periodic waveforms and also in cases when
    /// it does not depend on Time (in which case its frequency will be 0), for example, as a parameter
    /// of a Filter.
    Sine {
        frequency: Box<Waveform<MarkId, State>>,
        phase: Box<Waveform<MarkId, State>>,
        state: State,
    },
    /// Implements an impulse response filter with feed-forward and feedback coefficients. Assumes that
    /// the first feedback coefficient (a_0) is 1.0. If the filter has no feedback coefficients, then the
    /// filter has a finite response -- that is, it is a convolution.
    Filter {
        waveform: Box<Waveform<MarkId, State>>,
        feed_forward: Vec<Waveform<MarkId, State>>, // b_0, b_1, ...
        feedback: Vec<Waveform<MarkId, State>>,     // a_1, a_2, ...
        state: State,
    },
    /// Implements a point-wise transformation of two waveforms by computing the given operation one
    /// sample at a time.
    BinaryPointOp(
        Operator,
        Box<Waveform<MarkId, State>>,
        Box<Waveform<MarkId, State>>,
    ),
    /// Generates a repeating waveform that restarts the given waveform whenever the trigger waveform
    /// flips from negative values to positive values. Its length is determined by the trigger waveform.
    Reset {
        trigger: Box<Waveform<MarkId, State>>,
        waveform: Box<Waveform<MarkId, State>>,
        state: State,
    },
    /// Generates a waveform by alternating between two waveforms based on the sign of the trigger
    /// waveform. Its length is determined by the trigger waveform.
    Alt {
        trigger: Box<Waveform<MarkId, State>>,
        positive_waveform: Box<Waveform<MarkId, State>>,
        negative_waveform: Box<Waveform<MarkId, State>>,
    },
    /// Generates the same samples as the inner waveform. Tracker status updates will include a list of
    /// marked waveforms, along with their start times and durations, and so marked waveforms can be
    /// used to signal that a certain event will occur or has occurred. Marked waveforms can also be
    /// used with Command::Modify to create dynamically changing waveforms.
    Marked {
        id: MarkId,
        waveform: Box<Waveform<MarkId, State>>,
    },
    /// Generate the same samples as the inner waveform and also writes them to a file whose name begins
    /// with the given file stem.
    Captured {
        file_stem: String,
        waveform: Box<Waveform<MarkId, State>>,
    },
    /// Represents a previous version of a waveform.
    ///
    /// Cannot be used to generate samples but may be used as part of commands (like Modify).
    Prior,
}

impl<MarkId: Display, State> Display for Waveform<MarkId, State> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Waveform::*;
        match self {
            Const(value) => write!(f, "Const({})", value),
            Time(_) => write!(f, "Time"),
            Noise => write!(f, "Noise"),
            Fixed(samples, _) => {
                if samples.len() <= 10 {
                    write!(f, "Fixed({:?})", samples)
                } else {
                    write!(
                        f,
                        "Fixed([{}, ...], len={})",
                        samples[..10]
                            .iter()
                            .map(|x| format!("{}", x))
                            .collect::<Vec<String>>()
                            .join(", "),
                        samples.len()
                    )
                }
            }
            Fin { length, waveform } => {
                write!(f, "Fin({}, {})", length, waveform)
            }
            Append(a, b) => write!(f, "Append({}, {})", a, b),
            Sine {
                frequency, phase, ..
            } => write!(f, "Sine({}, {})", frequency, phase),
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => {
                let ff: Vec<String> = feed_forward.iter().map(|w| format!("{}", w)).collect();
                let fb: Vec<String> = feedback.iter().map(|w| format!("{}", w)).collect();
                write!(
                    f,
                    "Filter({}, [{}], [{}])",
                    waveform,
                    ff.join(", "),
                    fb.join(", ")
                )
            }
            BinaryPointOp(op, a, b) => {
                write!(f, "{:?}({}, {})", op, a, b)
            }
            Reset {
                trigger, waveform, ..
            } => {
                write!(f, "Reset({}, {})", trigger, waveform)
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
            Marked { id, waveform } => {
                write!(f, "Marked({}, {})", id, waveform)
            }
            Captured {
                file_stem,
                waveform,
            } => {
                write!(f, "Captured({}, {})", file_stem, waveform)
            }
            Prior => write!(f, "Prior"),
        }
    }
}

/// Replaces the state of `waveform` and every component with `state`.
pub fn initialize_state<M, S, T>(waveform: Waveform<M, S>, state: T) -> Waveform<M, T>
where
    T: Clone,
{
    use Waveform::*;
    match waveform {
        Const(value) => Const(value),
        Time(_) => Time(state),
        Noise => Noise,
        Fixed(samples, _) => Fixed(samples, state),
        Fin { length, waveform } => Fin {
            length: Box::new(initialize_state(*length, state.clone())),
            waveform: Box::new(initialize_state(*waveform, state)),
        },
        Append(a, b) => Append(
            Box::new(initialize_state(*a, state.clone())),
            Box::new(initialize_state(*b, state)),
        ),
        Sine {
            frequency, phase, ..
        } => Sine {
            frequency: Box::new(initialize_state(*frequency, state.clone())),
            phase: Box::new(initialize_state(*phase, state.clone())),
            state: state,
        },
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => Filter {
            waveform: Box::new(initialize_state(*waveform, state.clone())),
            feed_forward: feed_forward
                .into_iter()
                .map(|w| initialize_state(w, state.clone()))
                .collect(),
            feedback: feedback
                .into_iter()
                .map(|w| initialize_state(w, state.clone()))
                .collect(),
            state: state,
        },
        BinaryPointOp(op, a, b) => BinaryPointOp(
            op,
            Box::new(initialize_state(*a, state.clone())),
            Box::new(initialize_state(*b, state)),
        ),
        Reset {
            trigger, waveform, ..
        } => Reset {
            trigger: Box::new(initialize_state(*trigger, state.clone())),
            waveform: Box::new(initialize_state(*waveform, state.clone())),
            state: state,
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(initialize_state(*trigger, state.clone())),
            positive_waveform: Box::new(initialize_state(*positive_waveform, state.clone())),
            negative_waveform: Box::new(initialize_state(*negative_waveform, state)),
        },
        Marked { id, waveform } => Marked {
            id,
            waveform: Box::new(initialize_state(*waveform, state)),
        },
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(initialize_state(*waveform, state)),
        },
        Prior => Prior,
    }
}

/// Replaces the state of `waveform` and every component with the default (unit) state.
pub fn remove_state<M, S>(waveform: Waveform<M, S>) -> Waveform<M> {
    use Waveform::*;
    match waveform {
        Const(value) => Const(value),
        Time(_) => Time(()),
        Noise => Noise,
        Fixed(samples, _) => Fixed(samples, ()),
        Fin { length, waveform } => Fin {
            length: Box::new(remove_state(*length)),
            waveform: Box::new(remove_state(*waveform)),
        },
        Append(a, b) => Append(Box::new(remove_state(*a)), Box::new(remove_state(*b))),
        Sine {
            frequency, phase, ..
        } => Sine {
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
            feed_forward: feed_forward.into_iter().map(remove_state).collect(),
            feedback: feedback.into_iter().map(remove_state).collect(),
            state: (),
        },
        BinaryPointOp(op, a, b) => {
            BinaryPointOp(op, Box::new(remove_state(*a)), Box::new(remove_state(*b)))
        }
        Reset {
            trigger, waveform, ..
        } => Reset {
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
        Prior => Prior,
    }
}

/// Sets the state of `waveform` and every component to `new_state`.
pub fn set_state<M, S>(waveform: &mut Waveform<M, S>, new_state: S)
where
    S: Clone,
{
    use Waveform::*;
    match waveform {
        Const(_) => (),
        Time(state) => *state = new_state,
        Noise => (),
        Fixed(_, state) => *state = new_state,
        Fin { length, waveform } => {
            set_state(length, new_state.clone());
            set_state(waveform, new_state);
        }
        Append(a, b) => {
            set_state(a, new_state.clone());
            set_state(b, new_state);
        }
        Sine {
            frequency,
            phase,
            state,
        } => {
            set_state(frequency, new_state.clone());
            set_state(phase, new_state.clone());
            *state = new_state;
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => {
            set_state(waveform, new_state.clone());
            let _ = feed_forward
                .into_iter()
                .map(|w| set_state(w, new_state.clone()));
            let _ = feedback
                .into_iter()
                .map(|w| set_state(w, new_state.clone()));
            *state = new_state;
        }
        BinaryPointOp(_, a, b) => {
            set_state(a, new_state.clone());
            set_state(b, new_state);
        }
        Reset {
            trigger,
            waveform,
            state,
        } => {
            set_state(trigger, new_state.clone());
            set_state(waveform, new_state.clone());
            *state = new_state;
        }
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            set_state(trigger, new_state.clone());
            set_state(positive_waveform, new_state.clone());
            set_state(negative_waveform, new_state);
        }
        Marked { waveform, .. } => {
            set_state(waveform, new_state);
        }
        Captured { waveform, .. } => {
            set_state(waveform, new_state);
        }
        Prior => (),
    }
}

/// Replaces zero or more parts of `waveform` with a copy of `new_waveform.`
///
/// If `mark_id` is Some then it will replace the contents of all Marked waveforms whose
/// id matches. If `mark_id` is None, then it will replace all Prior waveforms.
pub fn substitute<M, S>(
    waveform: &mut Waveform<M, S>,
    mark_id: &Option<M>,
    new_waveform: &Waveform<M, S>,
) where
    S: Clone + Debug,
    M: Clone + PartialEq + Display + Debug,
{
    use Waveform::*;
    match waveform {
        Marked { id, waveform } => match mark_id {
            Some(m_id) => {
                if id == m_id {
                    let mut new_waveform = new_waveform.clone();
                    substitute(&mut new_waveform, &None, &waveform);
                    *waveform = Box::new(new_waveform);
                } else {
                    substitute(waveform, mark_id, new_waveform);
                }
            }
            None => {
                substitute(waveform, mark_id, new_waveform);
            }
        },
        Prior if *mark_id == None => {
            *waveform = new_waveform.clone();
        }
        Prior => (),
        // Leaf nodes — nothing to recurse into
        Const(_) | Time(_) | Noise | Fixed(..) => {}

        Fin { length, waveform } => {
            substitute(length, mark_id, new_waveform);
            substitute(waveform, mark_id, new_waveform);
        }
        Append(a, b) => {
            substitute(a, mark_id, new_waveform);
            substitute(b, mark_id, new_waveform);
        }
        Sine {
            frequency, phase, ..
        } => {
            substitute(frequency, mark_id, new_waveform);
            substitute(phase, mark_id, new_waveform);
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => {
            substitute(waveform, mark_id, new_waveform);
            for w in feed_forward {
                substitute(w, mark_id, new_waveform);
            }
            for w in feedback {
                substitute(w, mark_id, new_waveform);
            }
        }
        BinaryPointOp(_, a, b) => {
            substitute(a, mark_id, new_waveform);
            substitute(b, mark_id, new_waveform);
        }
        Reset {
            trigger, waveform, ..
        } => {
            substitute(trigger, mark_id, new_waveform);
            substitute(waveform, mark_id, new_waveform);
        }
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            substitute(trigger, mark_id, new_waveform);
            substitute(positive_waveform, mark_id, new_waveform);
            substitute(negative_waveform, mark_id, new_waveform);
        }
        Captured { waveform, .. } => substitute(waveform, mark_id, new_waveform),
    }
}
