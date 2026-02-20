use std::fmt;
use std::fmt::Debug;

// TODO move this out of the waveform?
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
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
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
        length: Box<Waveform<State>>,
        waveform: Box<Waveform<State>>,
    },
    /*
     * Seq sets the offset of `waveform`. The offset is determined by the first point at which the `offset` waveform
     * is >= 0.0. For example, `Seq(Const(0.0), _)` has an offset 0 seconds and `Seq(Subtract(Time, Const(2.0)), _)` has
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
     * Sin computes the sine with the given frequency and phase (both in radians). Or put another way,
     * it computes the sine of angle that changes according to the rate of the first parameter and the
     * value of the second. Note that Sin is used both as the basis for periodic waveforms and also in
     * cases when it does not depend on Time (in which case its frequency will be 0), for example, as
     * a parameter of a Filter.
     */
    Sin {
        frequency: Box<Waveform<State>>,
        phase: Box<Waveform<State>>,
        state: State,
    },
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
    BinaryPointOp(Operator, Box<Waveform<State>>, Box<Waveform<State>>),
    /*
     * Res generates a repeating waveform that restarts the given waveform whenever the trigger
     * waveform flips from negative values to positive values. Its length and offset are determined
     * by the trigger waveform.
     */
    Res {
        trigger: Box<Waveform<State>>,
        waveform: Box<Waveform<State>>,
        state: State,
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

impl<State> fmt::Display for Waveform<State> {
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
            Seq { offset, waveform } => {
                write!(f, "Seq({}, {})", offset, waveform)
            }
            Append(a, b) => write!(f, "Append({}, {})", a, b),
            Sin {
                frequency, phase, ..
            } => write!(f, "Sin({}, {})", frequency, phase),
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => write!(f, "Filter({}, {}, {})", waveform, feed_forward, feedback),
            BinaryPointOp(op, a, b) => {
                write!(f, "{:?}({}, {})", op, a, b)
            }
            Res {
                trigger, waveform, ..
            } => {
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

pub fn initialize_state<S, T>(waveform: Waveform<S>, state: T) -> Waveform<T>
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
        Seq { offset, waveform } => Seq {
            offset: Box::new(initialize_state(*offset, state.clone())),
            waveform: Box::new(initialize_state(*waveform, state)),
        },
        Append(a, b) => Append(
            Box::new(initialize_state(*a, state.clone())),
            Box::new(initialize_state(*b, state)),
        ),
        Sin {
            frequency, phase, ..
        } => Sin {
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
            // Don't initialize the inner waveform until we can determine the length of feed_forward
            waveform: Box::new(initialize_state(*waveform, state.clone())),
            feed_forward: Box::new(initialize_state(*feed_forward, state.clone())),
            feedback: Box::new(initialize_state(*feedback, state.clone())),
            state: state,
        },
        BinaryPointOp(op, a, b) => BinaryPointOp(
            op,
            Box::new(initialize_state(*a, state.clone())),
            Box::new(initialize_state(*b, state)),
        ),
        Res {
            trigger, waveform, ..
        } => Res {
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

pub fn remove_state<State>(w: Waveform<State>) -> Waveform<()> {
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
