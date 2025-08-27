use std::{iter::Sum, time::Duration};

use crate::tracker::Waveform;

enum OneOrTwo<T> {
    One(T),
    Two(T, T),
}

// Replaces all Seq's with Appends and returns the overall offset of the waveform.
fn replace_seq(waveform: Waveform) -> (Duration, Waveform) {
    use Waveform::*;
    match waveform {
        w @ (Const(_) | Time | Noise | Fixed(_)) => (Duration::ZERO, w),
        Fin { duration, waveform } => {
            // Offset of the duration waveform doesn't matter
            let (_, duration) = replace_seq(*duration);
            let (seq_duration, waveform) = replace_seq(*waveform);
            (seq_duration, Fin { duration: Box::new(duration), waveform: Box::new(waveform) })
        }
        Seq { duration, waveform } => {
            let (inner_duration, waveform) = replace_seq(*waveform);
            (duration + inner_duration, waveform)
        }

        Sin(waveform) => {
            let (duration, waveform) = replace_seq(*waveform);
            (duration, Sin(Box::new(waveform)))
        }

        Filter { waveform, feed_forward, feedback, state } => {
            let (duration, waveform) = replace_seq(*waveform);
            let (_, feed_forward) = replace_seq(*feed_forward);
            let (_, feedback) = replace_seq(*feedback);
            (duration, Filter { waveform: Box::new(waveform), feed_forward: Box::new(feed_forward), feedback: Box::new(feedback), state })
        }


        Sum(a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            if a_offset == Duration::ZERO {
                (b_offset, Sum(Box::new(a), Box::new(b)))
            } else {
                (a_offset + b_offset, Sum(Box::new(a), Box::new(Append(
                  Fin
                  Box::new(b) 
                ))))
            }
        }

    }
}

// Combines constants in binary operators
// No binary operation will contain two Consts
// Sin(Const|Fixed) => Const|Fixed
// XXX push Fin's down?
pub fn simplify(waveform: Waveform) -> Waveform {
    use OneOrTwo::*;
    use Waveform::*;
    match waveform {
        // No changes for these:
        w @ (Const(_) | Time | Noise | Fixed(_)) => w,
        // Check to see if the waveform has 0 length:
        Fin { duration, waveform } => {
            let duration = simplify(*duration);
            match duration {
                Const(a) if a >= 0.0 => Fixed(vec![]),
                Fixed(v) if v.len() >= 1 && v[0] >= 0.0 => Fixed(vec![]),
                duration => match simplify(*waveform) {
                    waveform => Fin {
                        duration: Box::new(duration),
                        waveform: Box::new(waveform),
                    },
                },
            }
        }
        // Check to see if the waveform has 0 offset or if there are nested Seq's:
        Seq { duration, waveform } => {
            let waveform = simplify(*waveform);
            match (duration, waveform) {
                (Duration::ZERO, waveform) => waveform,
                (
                    duration,
                    Seq {
                        duration: inner_duration,
                        waveform,
                    },
                ) => Seq {
                    duration: duration + inner_duration,
                    waveform,
                },
                (duration, waveform) => Seq {
                    duration,
                    waveform: Box::new(waveform),
                },
            }
        }
        // Check to see if we can precompute the sine function:
        Sin(waveform) => {
            let waveform = simplify(*waveform);
            match waveform {
                Const(a) => Const(a.sin()),
                Fixed(v) => {
                    let v = v.into_iter().map(|x| x.sin()).collect();
                    Fixed(v)
                }
                Seq { duration, waveform } => Seq {
                    duration,
                    waveform: Box::new(Sin(waveform)),
                },
                waveform => Sin(Box::new(waveform)),
            }
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => {
            let waveform = simplify(*waveform);
            let feed_forward = simplify(*feed_forward);
            let feedback = simplify(*feedback);
            Filter {
                waveform: Box::new(waveform),
                feed_forward: Box::new(feed_forward),
                feedback: Box::new(feedback),
                state,
            }
        }
        Sum(a, b) => match simplify_binary_op(*a, *b, |x, y| x + y) {
            One(a) => a,
            Two(Const(a), Sum(b, c)) => match (*b, *c) {
                (Const(b), c) => Sum(Box::new(Const(a + b)), Box::new(c)),
                (b, c) => Sum(Box::new(Const(a)), Box::new(Sum(Box::new(b), Box::new(c)))),
            },
            Two(Sum(a, b), Const(c)) => match (*a, *b) {
                (a, Const(b)) => Sum(Box::new(a), Box::new(Const(b + c))),
                (a, b) => Sum(Box::new(Sum(Box::new(a), Box::new(b))), Box::new(Const(c))),
            },
            Two(a, b) => Sum(Box::new(a), Box::new(b)),
        },
        DotProduct(a, b) => match simplify_binary_op(*a, *b, |x, y| x * y) {
            One(a) => a,
            Two(Const(a), DotProduct(b, c)) => match (*b, *c) {
                (Const(b), c) => DotProduct(Box::new(Const(a * b)), Box::new(c)),
                (b, c) => DotProduct(
                    Box::new(Const(a)),
                    Box::new(DotProduct(Box::new(b), Box::new(c))),
                ),
            },
            Two(DotProduct(a, b), Const(c)) => match (*a, *b) {
                (a, Const(b)) => DotProduct(Box::new(a), Box::new(Const(b * c))),
                (a, b) => DotProduct(
                    Box::new(DotProduct(Box::new(a), Box::new(b))),
                    Box::new(Const(c)),
                ),
            },
            Two(a, b) => DotProduct(Box::new(a), Box::new(b)),
        },
        Res { trigger, waveform } => {
            let trigger = simplify(*trigger);
            let waveform = simplify(*waveform);
            Res {
                trigger: Box::new(trigger),
                waveform: Box::new(waveform),
            }
        }
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            let trigger = simplify(*trigger);
            let positive_waveform = simplify(*positive_waveform);
            let negative_waveform = simplify(*negative_waveform);
            Alt {
                trigger: Box::new(trigger),
                positive_waveform: Box::new(positive_waveform),
                negative_waveform: Box::new(negative_waveform),
            }
        }
        w @ Slider(_) => w,
        Marked { id, waveform } => {
            let waveform = simplify(*waveform);
            Marked {
                id,
                waveform: Box::new(waveform),
            }
        }
        Captured {
            file_stem,
            waveform,
        } => {
            let waveform = simplify(*waveform);
            Captured {
                file_stem,
                waveform: Box::new(waveform),
            }
        }
    }
}

// Simplifies each of `a` and `b`, then combines if possible, otherwise returns to operands that should be combined
// with the original waveform operator.
fn simplify_binary_op(a: Waveform, b: Waveform, op: fn(f32, f32) -> f32, variant: fn(Box<Waveform>, Box<Waveform>) -> Waveform) -> OneOrTwo<Waveform> {
    use OneOrTwo::*;
    use Waveform::*;
    let a = simplify(a);
    let b = simplify(b);
    match (a, b) {
        (Const(a), Const(b)) => One(Const(op(a, b))),
        // Push down into Append
        (a, Append(b, c)) => 
            One(Append(simplify(variant(Box::new(b), Box::new(a))),
                simplify(variant(Box::new(c), Box::new(a))))),

        (Append(b, c), a) => 
            One(Append(simplify(variant(Box::new(b), Box::new(a))),
                simplify(variant(Box::new(c), Box::new(a))))),

        


        (Const(a), Seq { duration, waveform })  => match *waveform {
            Const(b) => One(Seq {
                duration,
                waveform: Box::new(Const(op(a, b))),
            }),
            _ => Two(
                // XXX need to return Seq(d, Op(Const(a), waveform))
                Const(a),
                Seq {
                    duration,
                    waveform: Box::new(*waveform),
                },
            ),
        },
        // XXX need to handle (Seq(d, a), Seq(e, b))
        //   ==> Seq(d+e, Op(a, Delay(d, b)))
        // XXX need to handle (Seq(d, a), b)) for non-seq, b
        //  ==> Seq(d, Op(a, Delay(d, b)))
        // XXX need to handle (a, Seq(d, b)) for non-const a, non-seq a
        //  ==> Seq(d, Op(a, b))
        (a, b) => Two(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Waveform::*;

    #[test]
    fn test_simplify() {
        let w1 = Sum(
            Box::new(Sum(
                Box::new(Const(1.0)),
                Box::new(Sum(Box::new(Const(2.0)), Box::new(Const(3.0)))),
            )),
            Box::new(Const(4.0)),
        );
        assert_eq!(simplify(w1), Const(10.0));

        let w2 = Sum(
            Box::new(Sum(
                Box::new(Const(2.0)),
                Box::new(Sum(Box::new(Const(3.0)), Box::new(Sin(Box::new(Time))))),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w2),
            Sum(Box::new(Const(10.0)), Box::new(Sin(Box::new(Time))),),
        );

        let w3 = DotProduct(
            Box::new(DotProduct(
                Box::new(Const(2.0)),
                Box::new(DotProduct(
                    Box::new(Const(3.0)),
                    Box::new(Sin(Box::new(Time))),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w3),
            DotProduct(Box::new(Const(30.0)), Box::new(Sin(Box::new(Time)))),
        );

        let w4 = Seq {
            duration: Duration::ZERO,
            waveform: Box::new(Sin(Box::new(Time))),
        };
        assert_eq!(simplify(w4), Sin(Box::new(Time)));

        // Nested Seq
        let w5 = Seq {
            duration: Duration::from_secs_f32(1.0),
            waveform: Box::new(Seq {
                duration: Duration::from_secs_f32(2.0),
                waveform: Box::new(Sin(Box::new(Time))),
            }),
        };
        assert_eq!(
            simplify(w5),
            Seq {
                duration: Duration::from_secs_f32(3.0),
                waveform: Box::new(Sin(Box::new(Time))),
            }
        );
    }
}
