use crate::waveform::{Operator, Waveform};

// First root returns the first non-negative value at which the given waveform is zero. This
// is implemented for waveforms of the forms:
//   * BinaryPointOp(Operator::Add|Operator::Subtract, Time, _)
//   * Time
//   * Const(0)
// It returns None otherwise.
pub fn first_root<M>(waveform: &Waveform<M>) -> Option<Waveform<M>>
where
    M: Clone + PartialEq,
{
    use Waveform::*;
    match waveform {
        Const(0.0) => Some(Const(0.0)),
        Const(_) => None,
        Time(_) => Some(Const(0.0)),
        BinaryPointOp(Operator::Add, a, b) => match (&**a, &**b) {
            // TODO should really check that Time doesn't appear on the other side too
            (Time(_), w) => Some(optimize(BinaryPointOp(
                Operator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            (w, Time(_)) => Some(optimize(BinaryPointOp(
                Operator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            _ => None,
        },
        BinaryPointOp(Operator::Subtract, a, b) => first_root(&BinaryPointOp(
            Operator::Add,
            a.clone(),
            Box::new(optimize(BinaryPointOp(
                Operator::Multiply,
                b.clone(),
                Box::new(Const(-1.0)),
            ))),
        )),
        _ => None,
    }
}

// Optimize waveform expressions by...
//   * eliminating constants in binary operators and Sine
//   * re-associates binary operations so that Consts are on the right
//   * pulling Fin's up and combining nested Fin's
//   * replacing zero-length waveforms with the canonical `Fixed(vec![])`
//   * handling some common cases of Fin in Sum and DotProduct
// This function must be called after replace_seq.
pub fn optimize<M>(waveform: Waveform<M>) -> Waveform<M>
where
    M: Clone + PartialEq,
{
    use Waveform::*;
    match waveform {
        // No changes for these:
        w @ (Const(_) | Time(_) | Noise | Fixed(_, _)) => w,
        Fin { length, waveform } => {
            let length = optimize(*length);
            match length {
                // Zero length
                Const(a) if a >= 0.0 => Fixed(vec![], ()),
                Fixed(v, _) if v.len() >= 1 && v[0] >= 0.0 => Fixed(vec![], ()),
                // TODO for longer Fixed, replace with * Fixed(vec![1.0; v.len()])?
                Time(_) => Fixed(vec![], ()),
                length => match optimize(*waveform) {
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                    } => match (first_root(&length), first_root(&*inner_length)) {
                        (Some(Const(a)), Some(Const(b))) => Fin {
                            length: Box::new(optimize(BinaryPointOp(
                                Operator::Subtract,
                                Box::new(Time(())),
                                Box::new(Const(a.min(b))),
                            ))),
                            waveform,
                        },
                        _ => Fin {
                            length: Box::new(length),
                            waveform: Box::new(Fin {
                                length: inner_length,
                                waveform,
                            }),
                        },
                    },
                    waveform => Fin {
                        length: Box::new(length),
                        waveform: Box::new(waveform),
                    },
                },
            }
        }
        Append(a, b) => {
            let a = optimize(*a);
            let b = optimize(*b);
            match (a, b) {
                (Fixed(a, _), b) if a.len() == 0 => b,
                (a, Fixed(b, _)) if b.len() == 0 => a,
                (Fixed(a, _), Fixed(b, _)) => Fixed([a, b].concat(), ()),
                (a, b) => Append(Box::new(a), Box::new(b)),
            }
        }
        // Check to see if we can compute the sine function:
        Sine {
            frequency,
            phase,
            state,
        } => {
            let frequency = optimize(*frequency);
            let phase = optimize(*phase);
            match (frequency, phase) {
                (Const(0.0), Const(p)) => Const(p.sin()),
                (Const(0.0), Fixed(v, _)) => {
                    let v = v.into_iter().map(|x| x.sin()).collect();
                    Fixed(v, ())
                }
                (frequency, phase) => Sine {
                    frequency: Box::new(frequency),
                    phase: Box::new(phase),
                    state,
                },
            }
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => Filter {
            waveform: Box::new(optimize(*waveform)),
            feed_forward: feed_forward.into_iter().map(optimize).collect(),
            feedback: feedback.into_iter().map(optimize).collect(),
            state,
        },
        BinaryPointOp(Operator::Add, a, b) => {
            match (optimize(*a), optimize(*b)) {
                // Add yields the shorter of the two inputs.
                (Fixed(a, _), _) if a.len() == 0 => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.len() == 0 => Fixed(vec![], ()),
                (Const(a), Const(b)) => Const(a + b),
                // Adding 0 is identity (because Add truncates to shorter, and Const is infinite)
                (a, Const(0.0)) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryPointOp(
                    Operator::Add,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                (BinaryPointOp(Operator::Add, a, b), Const(c)) => BinaryPointOp(
                    Operator::Add,
                    a,
                    Box::new(optimize(BinaryPointOp(
                        Operator::Add,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // TODO could distribute constants over Append(Fin, _), Reset, and Alt
                // ... though Alt generates both branches, so better not to do too much work

                // Combine two Fins with the same length
                (
                    Fin {
                        length: a_length,
                        waveform: a,
                    },
                    Fin {
                        length: b_length,
                        waveform: b,
                    },
                ) if first_root(&a_length) == first_root(&b_length) => Fin {
                    length: a_length,
                    waveform: Box::new(optimize(BinaryPointOp(Operator::Add, a, b))),
                },
                (a, b) => BinaryPointOp(Operator::Add, Box::new(a), Box::new(b)),
            }
        }
        BinaryPointOp(Operator::Subtract, a, b) => optimize(BinaryPointOp(
            Operator::Add,
            a,
            Box::new(optimize(BinaryPointOp(
                Operator::Multiply,
                b,
                Box::new(Const(-1.0)),
            ))),
        )),
        BinaryPointOp(Operator::Merge, a, b) => {
            match (optimize(*a), optimize(*b)) {
                // Merge yields the longer of the two inputs.
                (Fixed(a, _), b) if a.len() == 0 => b,
                (a, Fixed(b, _)) if b.len() == 0 => a,
                (Const(a), Const(b)) => Const(a + b),
                // Merging 0 is the identity if the left-hand side is infinite
                // TODO could check for other infinite waveforms
                (a @ (Time(_) | Noise), Const(0.0)) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryPointOp(
                    Operator::Merge,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Combine merge of Fin and an Append who first argument is Fin -- this occurs for expressions of
                // the form `w | fin(t) | seq(t)`.
                (
                    Fin {
                        length: a_length,
                        waveform: a,
                    },
                    Append(b, c),
                ) => match *b {
                    Fin {
                        length: b_length,
                        waveform: b,
                    } if first_root(&a_length) == first_root(&b_length) => optimize(Append(
                        Box::new(Fin {
                            length: a_length,
                            waveform: Box::new(BinaryPointOp(Operator::Merge, a, b)),
                        }),
                        c,
                    )),
                    _ => BinaryPointOp(
                        Operator::Merge,
                        Box::new(Fin {
                            length: a_length,
                            waveform: a,
                        }),
                        Box::new(Append(b, c)),
                    ),
                },
                (a, b) => BinaryPointOp(Operator::Merge, Box::new(a), Box::new(b)),
            }
        }
        BinaryPointOp(Operator::Multiply, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (Fixed(a, _), _) if a.len() == 0 => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.len() == 0 => Fixed(vec![], ()),
                (Const(1.0), b) => b,
                (a, Const(1.0)) => a,
                (Const(a), Const(b)) => Const(a * b),
                (Fixed(a, _), Const(b)) => Fixed(a.into_iter().map(|x| x * b).collect(), ()),
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                (BinaryPointOp(Operator::Multiply, a, b), Const(c)) => BinaryPointOp(
                    Operator::Multiply,
                    a,
                    Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // Distribute
                // (a + b) * c == (a * c) + (b * c)
                (BinaryPointOp(Operator::Add, a, b), Const(c)) => BinaryPointOp(
                    Operator::Add,
                    Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        a,
                        Box::new(Const(c)),
                    ))),
                    Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // (a / b) * c == (a * c) / b
                (BinaryPointOp(Operator::Divide, a, b), Const(c)) => BinaryPointOp(
                    Operator::Divide,
                    Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        a,
                        Box::new(Const(c)),
                    ))),
                    b,
                ),
                // TODO could check the inside of Marked/Capture.

                // TODO could distribute constants over, Append, Reset, and Alt
                // ... though currently Alt generates both branches, so better not to do too much work

                // Pull Fin out
                (Fin { length, waveform }, b) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryPointOp(
                        Operator::Multiply,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryPointOp(Operator::Multiply, Box::new(a), Box::new(b)),
            }
        }
        BinaryPointOp(Operator::Divide, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (_, Fixed(b, _)) if b.len() == 0 => Fixed(vec![], ()),
                // Prefer multiplication
                (a, Const(b)) => optimize(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(a),
                    Box::new(Const(1.0 / b)),
                )),
                // ((a / b) / c) == (a / (b * c))
                (BinaryPointOp(Operator::Divide, a, b), c) => BinaryPointOp(
                    Operator::Divide,
                    a,
                    Box::new(optimize(BinaryPointOp(Operator::Multiply, b, Box::new(c)))),
                ),
                // (a / (b / c)) == (a * c) / b
                (a, BinaryPointOp(Operator::Divide, b, c)) => BinaryPointOp(
                    Operator::Divide,
                    Box::new(optimize(BinaryPointOp(Operator::Multiply, Box::new(a), c))),
                    b,
                ),

                // Pull Fin out
                (Fin { length, waveform }, b) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryPointOp(
                        Operator::Divide,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryPointOp(
                        Operator::Divide,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryPointOp(Operator::Divide, Box::new(a), Box::new(b)),
            }
        }
        Reset {
            trigger,
            waveform,
            state,
        } => Reset {
            trigger: Box::new(optimize(*trigger)),
            waveform: Box::new(optimize(*waveform)),
            state,
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(optimize(*trigger)),
            positive_waveform: Box::new(optimize(*positive_waveform)),
            negative_waveform: Box::new(optimize(*negative_waveform)),
        },
        w @ Slider(_) => w,
        Marked { id, waveform } => {
            // TODO could pull out Fin if process_marks better implemented Fin
            Marked {
                id,
                waveform: Box::new(optimize(*waveform)),
            }
        }
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(optimize(*waveform)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Waveform::*;

    #[test]
    fn test_optimize() {
        let w1: Waveform<(), ()> = BinaryPointOp(
            Operator::Add,
            Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(1.0)),
                Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Const(2.0)),
                    Box::new(Const(3.0)),
                )),
            )),
            Box::new(Const(4.0)),
        );
        assert_eq!(optimize(w1), Const(10.0));

        let w2: Waveform<(), ()> = BinaryPointOp(
            Operator::Add,
            Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Const(3.0)),
                    Box::new(Sine {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w2),
            BinaryPointOp(
                Operator::Add,
                Box::new(Sine {
                    frequency: Box::new(Const(1.0)),
                    phase: Box::new(Const(0.0)),
                    state: ()
                }),
                Box::new(Const(10.0))
            ),
        );

        let w3: Waveform<(), ()> = BinaryPointOp(
            Operator::Multiply,
            Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Sine {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w3),
            BinaryPointOp(
                Operator::Multiply,
                Box::new(Sine {
                    frequency: Box::new(Const(1.0)),
                    phase: Box::new(Const(0.0)),
                    state: ()
                }),
                Box::new(Const(30.0))
            ),
        );

        let w4: Waveform<(), ()> = BinaryPointOp(
            Operator::Multiply,
            Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Sine {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w4),
            BinaryPointOp(
                Operator::Add,
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Sine {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: ()
                    }),
                    Box::new(Const(15.0))
                )),
                Box::new(Const(10.0))
            ),
        );

        let w5: Waveform<(), ()> = BinaryPointOp(
            Operator::Multiply,
            Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-2.0)),
                )),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-1.5)),
                )),
                waveform: Box::new(Const(5.0)),
            }),
        );
        assert_eq!(
            optimize(w5),
            Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-1.5))
                )),
                waveform: Box::new(Const(15.0)),
            }
        );
    }
}
