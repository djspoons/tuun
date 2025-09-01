use crate::tracker::Waveform;

fn first_root(waveform: &Waveform) -> Option<f32> {
    use Waveform::*;
    match waveform {
        Const(0.0) => Some(0.0),
        Const(_) => None,
        Time => Some(0.0),
        Sum(a, b) => match (&**a, &**b) {
            (Time, Const(x)) => Some(-x),
            (Const(x), Time) => Some(-x),
            _ => None,
        },
        _ => None,
    }
}

// Assume offset waveforms are of the form `Time ~+ Const(x)` or `Const(x)`. (We can relax to other linear
// functions of Time later.)
fn add_offsets(a: Waveform, b: Waveform) -> Waveform {
    use Waveform::*;
    match (first_root(&a), first_root(&b)) {
        (Some(a_root), Some(b_root)) => {
            let new_root = a_root + b_root;
            Sum(Box::new(Time), Box::new(Const(-new_root)))
        }
        (a_root, b_root) => {
            panic!(
                "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
                a_root, b_root, a, b
            );
        }
    }
}

// Replaces all Seq's with Appends and returns the overall offset of the waveform.
pub fn replace_seq(waveform: Waveform) -> (Waveform, Waveform) {
    use Waveform::*;
    match waveform {
        w @ (Const(_) | Time | Noise | Fixed(_)) => (Const(0.0), w),
        Fin { length, waveform } => {
            // Offset of the length waveform doesn't matter
            let (_, length) = replace_seq(*length);
            let (offset, waveform) = replace_seq(*waveform);
            (
                offset,
                Fin {
                    length: Box::new(length),
                    waveform: Box::new(waveform),
                },
            )
        }
        Seq { offset, waveform } => {
            // The offset of the offset waveform doesn't matter
            let (_, offset) = replace_seq(*offset);
            // Seq ignores the offset of the inner waveform
            let (_, waveform) = replace_seq(*waveform);
            (offset, waveform)
        }
        Append(a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            (
                add_offsets(a_offset, b_offset),
                Append(Box::new(a), Box::new(b)),
            )
        }
        Sin(waveform) => {
            let (offset, waveform) = replace_seq(*waveform);
            (offset, Sin(Box::new(waveform)))
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => {
            let (offset, waveform) = replace_seq(*waveform);
            (
                offset,
                Filter {
                    waveform: Box::new(waveform),
                    feed_forward: Box::new(replace_seq(*feed_forward).1),
                    feedback: Box::new(replace_seq(*feedback).1),
                    state,
                },
            )
        }
        Sum(a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            if a_offset == Const(0.0) {
                return (b_offset, Sum(Box::new(a), Box::new(b)));
            }
            (
                add_offsets(a_offset.clone(), b_offset),
                Sum(
                    Box::new(a),
                    Box::new(Append(
                        Box::new(Fin {
                            length: Box::new(a_offset),
                            waveform: Box::new(Const(0.0)),
                        }),
                        Box::new(b),
                    )),
                ),
            )
        }
        DotProduct(a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            if a_offset == Const(0.0) {
                return (b_offset, DotProduct(Box::new(a), Box::new(b)));
            }
            (
                add_offsets(a_offset.clone(), b_offset),
                DotProduct(
                    Box::new(a),
                    Box::new(Append(
                        Box::new(Fin {
                            length: Box::new(a_offset),
                            waveform: Box::new(Const(1.0)),
                        }),
                        Box::new(b),
                    )),
                ),
            )
        }
        Res { trigger, waveform } => {
            let (offset, trigger) = replace_seq(*trigger);
            (
                offset,
                Res {
                    trigger: Box::new(trigger),
                    waveform: Box::new(replace_seq(*waveform).1),
                },
            )
        }
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            let (offset, trigger) = replace_seq(*trigger);
            (
                offset,
                Alt {
                    trigger: Box::new(trigger),
                    positive_waveform: Box::new(replace_seq(*positive_waveform).1),
                    negative_waveform: Box::new(replace_seq(*negative_waveform).1),
                },
            )
        }
        Slider(slider) => (Const(0.0), Slider(slider)),
        Marked { id, waveform } => {
            let (offset, waveform) = replace_seq(*waveform);
            (
                offset,
                Marked {
                    id,
                    waveform: Box::new(waveform),
                },
            )
        }
        Captured {
            file_stem,
            waveform,
        } => {
            let (offset, waveform) = replace_seq(*waveform);
            (
                offset,
                Captured {
                    file_stem,
                    waveform: Box::new(waveform),
                },
            )
        }
    }
}
