use std::fmt;

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
pub enum Waveform<FilterState = (), SinState = (), ResState = ()> {
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
     * is 0 seconds in length and `Fin(Subtract(Time, Const(2.0)), _)` is 2 seconds in length.
     */
    Fin {
        length: Box<Waveform<FilterState, SinState, ResState>>,
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
    },
    /*
     * Seq sets the offset of `waveform`. The offset is determined by the first point at which the `offset` waveform
     * is >= 0.0. For example, `Seq(Const(0.0), _)` has an offset 0 seconds and `Seq(Subtract(Time, Const(2.0)), _)` has
     * an offset of 2 seconds.
     */
    Seq {
        offset: Box<Waveform<FilterState, SinState, ResState>>,
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
    },
    /*
     * Append concatenates two waveforms, generating all samples from the first waveform and
     * then all samples from the second (regardless of the offset of the first).
     */
    Append(
        Box<Waveform<FilterState, SinState, ResState>>,
        Box<Waveform<FilterState, SinState, ResState>>,
    ),
    /*
     * Sin computes the sine of each sample in the given waveform. Note that Sin is used both as the basis for periodic
     * waveforms (in which case its argument will depend on Time) and as a general-purpose sine function, for example,
     * as a parameter of a Filter (in which it will not depend on Time).
     */
    Sin(Box<Waveform<FilterState, SinState, ResState>>, SinState),
    /*
     * Filter implements an impulse response filter with feed-forward and feedback coefficients. Assumes that the first
     * feedback coefficient (a_0) is 1.0. If the filter has no feedback coefficients, then the filter has a finite
     * response -- that is, it is a convolution.
     */
    // TODO maybe add a_0 back in?
    Filter {
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
        feed_forward: Box<Waveform<FilterState, SinState, ResState>>, // b_0, b_1, ...
        feedback: Box<Waveform<FilterState, SinState, ResState>>,     // a_1, a_2, ...
        state: FilterState,
    },
    BinaryPointOp(
        Operator,
        Box<Waveform<FilterState, SinState, ResState>>,
        Box<Waveform<FilterState, SinState, ResState>>,
    ),
    /*
     * Res generates a repeating waveform that restarts the given waveform whenever the trigger
     * waveform flips from negative values to positive values. Its length and offset are determined
     * by the trigger waveform.
     */
    Res {
        trigger: Box<Waveform<FilterState, SinState, ResState>>,
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
        state: ResState,
    },
    /*
     * Alt generates a waveform by alternating between two waveforms based on the sign of
     * the trigger waveform.
     */
    Alt {
        trigger: Box<Waveform<FilterState, SinState, ResState>>,
        positive_waveform: Box<Waveform<FilterState, SinState, ResState>>,
        negative_waveform: Box<Waveform<FilterState, SinState, ResState>>,
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
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
    },
    /* Captured waveforms generate the same samples as the inner waveform and also write them to a file
     * beginning with the given file stem.
     */
    Captured {
        file_stem: String,
        waveform: Box<Waveform<FilterState, SinState, ResState>>,
    },
}

impl fmt::Display for Waveform<()> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Waveform::*;
        match self {
            Const(value) => write!(f, "Const({})", value),
            Time => write!(f, "Time"),
            Noise => write!(f, "Noise"),
            Fixed(samples) => {
                if samples.len() < 10 {
                    write!(f, "Fixed({:?})", samples)
                } else {
                    write!(
                        f,
                        "Fixed([{}, ...], len={})",
                        samples[..9]
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
            Sin(waveform, _) => write!(f, "Sin({})", waveform),
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
