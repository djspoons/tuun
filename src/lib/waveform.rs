use std::fmt;
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Merge,
}

/// Waveform is a compact representation of a sequence of samples.
///
/// With its default type parameters, it represents abstract waveforms that may be parsed or displayed.
/// Implementations that generate samples may specialize the State parameter to manage internal state.
/// Implementation that use marks for other internal purposes may specialize the MarkId parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Waveform<State = (), MarkId = u32> {
    /*
     * Const produces a stream of samples where each sample is the same constant value.
     */
    Const(f32),
    /*
     * Time generates a stream based on the elapsed time from the start of the waveform, in seconds.
     */
    Time(State),
    /*
     * Noise generates random samples.
     */
    Noise,
    /*
     * Fixed generates the same, finite, sequence of samples.
     */
    Fixed(Vec<f32>, State),
    /*
     * Fin generates a finite waveform, truncating the underlying waveform. The length is determined
     * by the first point at which the `length` waveform is >= 0.0. For example, `Fin(Const(0.0), _)`
     * is 0 seconds in length and `Fin(Subtract(Time, Const(2.0)), _)` is 2 seconds in length.
     */
    Fin {
        length: Box<Waveform<State, MarkId>>,
        waveform: Box<Waveform<State, MarkId>>,
    },
    /*
     * Append concatenates two waveforms, generating all samples from the first waveform and
     * then all samples from the second (regardless of the offset of the first).
     */
    Append(Box<Waveform<State, MarkId>>, Box<Waveform<State, MarkId>>),
    /*
     * Sine computes the sine with the given frequency and phase (both in radians). Or put another way,
     * it computes the sine of angle that changes according to the rate of the first parameter and the
     * value of the second. Note that Sine is used both as the basis for periodic waveforms and also in
     * cases when it does not depend on Time (in which case its frequency will be 0), for example, as
     * a parameter of a Filter.
     */
    Sine {
        frequency: Box<Waveform<State, MarkId>>,
        phase: Box<Waveform<State, MarkId>>,
        state: State,
    },
    /*
     * Filter implements an impulse response filter with feed-forward and feedback coefficients. Assumes that the first
     * feedback coefficient (a_0) is 1.0. If the filter has no feedback coefficients, then the filter has a finite
     * response -- that is, it is a convolution.
     */
    // TODO maybe add a_0 back in?
    Filter {
        waveform: Box<Waveform<State, MarkId>>,
        feed_forward: Vec<Waveform<State, MarkId>>, // b_0, b_1, ...
        feedback: Vec<Waveform<State, MarkId>>,     // a_1, a_2, ...
        state: State,
    },
    BinaryPointOp(
        Operator,
        Box<Waveform<State, MarkId>>,
        Box<Waveform<State, MarkId>>,
    ),
    /*
     * Reset generates a repeating waveform that restarts the given waveform whenever the trigger
     * waveform flips from negative values to positive values. Its length and offset are determined
     * by the trigger waveform.
     */
    Reset {
        trigger: Box<Waveform<State, MarkId>>,
        waveform: Box<Waveform<State, MarkId>>,
        state: State,
    },
    /*
     * Alt generates a waveform by alternating between two waveforms based on the sign of
     * the trigger waveform.
     */
    Alt {
        trigger: Box<Waveform<State, MarkId>>,
        positive_waveform: Box<Waveform<State, MarkId>>,
        negative_waveform: Box<Waveform<State, MarkId>>,
    },
    /*
     * Slider generates samples from a named interactive "slider" input.
     */
    Slider(String),
    /*
     * Marked waveforms generate the same samples as the inner waveform and are used to signal that a certain
     * event will occur or has occurred. Each status update will include a list of marked waveforms, along with
     * their start times and durations.
     */
    Marked {
        id: MarkId,
        waveform: Box<Waveform<State, MarkId>>,
    },
    /* Captured waveforms generate the same samples as the inner waveform and also write them to a file
     * beginning with the given file stem.
     */
    Captured {
        file_stem: String,
        waveform: Box<Waveform<State, MarkId>>,
    },
}

impl<State, MarkId: fmt::Display> fmt::Display for Waveform<State, MarkId> {
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
            Slider(slider) => write!(f, "Slider(\"{}\")", slider),
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

pub fn initialize_state<S, T, M>(waveform: Waveform<S, M>, state: T) -> Waveform<T, M>
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
        Slider(slider) => Slider(slider),
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
    }
}

pub fn remove_state<S, M>(w: Waveform<S, M>) -> Waveform<(), M> {
    use Waveform::*;
    match w {
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

pub fn set_state<S, M>(waveform: &mut Waveform<S, M>, new_state: S)
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
        Slider(_) => (),
        Marked { waveform, .. } => {
            set_state(waveform, new_state);
        }
        Captured { waveform, .. } => {
            set_state(waveform, new_state);
        }
    }
}

// Replace the contents of any Marked waveform in `waveform` with the given mark_id with a copy of `new_waveform.`
pub fn substitute<S, M>(waveform: &mut Waveform<S, M>, mark_id: &M, new_waveform: &Waveform<S, M>)
where
    S: Clone,
    M: Clone + PartialEq + fmt::Display,
{
    use Waveform::*;
    match waveform {
        Marked { id, waveform } if id == mark_id => {
            *waveform = Box::new(new_waveform.clone());
        }
        // Recurse into all variants that contain child waveforms
        Marked { waveform, .. } => substitute(waveform, mark_id, new_waveform),
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
        // Leaf nodes — nothing to recurse into
        Const(_) | Time(_) | Noise | Fixed(..) | Slider(_) => {}
    }
}

// Transform all mark IDs in a waveform from type M to type N using the given function.
pub fn map_marks<S, M, N, F>(waveform: Waveform<S, M>, f: &F) -> Waveform<S, N>
where
    F: Fn(M) -> N,
    M: fmt::Display,
    N: fmt::Display,
{
    use Waveform::*;
    match waveform {
        Const(value) => Const(value),
        Time(state) => Time(state),
        Noise => Noise,
        Fixed(samples, state) => Fixed(samples, state),
        Fin { length, waveform } => Fin {
            length: Box::new(map_marks(*length, f)),
            waveform: Box::new(map_marks(*waveform, f)),
        },
        Append(a, b) => Append(Box::new(map_marks(*a, f)), Box::new(map_marks(*b, f))),
        Sine {
            frequency,
            phase,
            state,
        } => Sine {
            frequency: Box::new(map_marks(*frequency, f)),
            phase: Box::new(map_marks(*phase, f)),
            state,
        },
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => Filter {
            waveform: Box::new(map_marks(*waveform, f)),
            feed_forward: feed_forward.into_iter().map(|w| map_marks(w, f)).collect(),
            feedback: feedback.into_iter().map(|w| map_marks(w, f)).collect(),
            state,
        },
        BinaryPointOp(op, a, b) => {
            BinaryPointOp(op, Box::new(map_marks(*a, f)), Box::new(map_marks(*b, f)))
        }
        Reset {
            trigger,
            waveform,
            state,
        } => Reset {
            trigger: Box::new(map_marks(*trigger, f)),
            waveform: Box::new(map_marks(*waveform, f)),
            state,
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(map_marks(*trigger, f)),
            positive_waveform: Box::new(map_marks(*positive_waveform, f)),
            negative_waveform: Box::new(map_marks(*negative_waveform, f)),
        },
        Slider(slider) => Slider(slider),
        Marked { id, waveform } => Marked {
            id: f(id),
            waveform: Box::new(map_marks(*waveform, f)),
        },
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(map_marks(*waveform, f)),
        },
    }
}
