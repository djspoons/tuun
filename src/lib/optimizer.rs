use crate::waveform::{BinaryOperator, Waveform};

// First root returns the first non-negative value at which the given waveform is zero. This
// is implemented for waveforms of the forms:
//   * BinaryOp(BinaryOperator::Add|BinaryOperator::Subtract, Time, _)
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
        BinaryOp(BinaryOperator::Add, a, b) => match (&**a, &**b) {
            // TODO should really check that Time doesn't appear on the other side too
            (Time(_), w) => Some(optimize(BinaryOp(
                BinaryOperator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            (w, Time(_)) => Some(optimize(BinaryOp(
                BinaryOperator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            _ => None,
        },
        BinaryOp(BinaryOperator::Subtract, a, b) => first_root(&BinaryOp(
            BinaryOperator::Add,
            a.clone(),
            Box::new(optimize(BinaryOp(
                BinaryOperator::Multiply,
                b.clone(),
                Box::new(Const(-1.0)),
            ))),
        )),
        _ => None,
    }
}

// Optimize waveform expressions by...
//   * eliminating constants in binary and unary operators and Phase
//   * re-associates binary operations so that Consts are on the right
//   * pulling Fin's up and combining nested Fin's
//   * replacing zero-length waveforms with the canonical `Fixed(vec![])`
//   * handling some common cases of Fin in Sum and DotProduct
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
                Fixed(v, _) if !v.is_empty() && v[0] >= 0.0 => Fixed(vec![], ()),
                // TODO for longer Fixed, replace with * Fixed(vec![1.0; v.len()])?
                Time(_) => Fixed(vec![], ()),
                length => match optimize(*waveform) {
                    // Truncating an empty waveform leaves it empty.
                    Fixed(v, _) if v.is_empty() => Fixed(vec![], ()),
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                    } => match (first_root(&length), first_root(&*inner_length)) {
                        (Some(Const(a)), Some(Const(b))) => Fin {
                            length: Box::new(optimize(BinaryOp(
                                BinaryOperator::Subtract,
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
        Append(a, b, state) => {
            let a = optimize(*a);
            let b = optimize(*b);
            match (a, b) {
                (Fixed(a, _), b) if a.is_empty() => b,
                (a, Fixed(b, _)) if b.is_empty() => a,
                (Fixed(a, _), Fixed(b, _)) => Fixed([a, b].concat(), ()),
                (a, b) => Append(Box::new(a), Box::new(b), state),
            }
        }
        Phase {
            frequency,
            offset,
            state,
        } => {
            let frequency = optimize(*frequency);
            let offset = optimize(*offset);
            match (frequency, offset) {
                (Const(0.0), Const(o)) => {
                    // Match the generator: wrap in f64, and keep the range
                    // half-open if rounding to f32 gives 1.0.
                    let p = (o as f64).rem_euclid(1.0) as f32;
                    Const(if p >= 1.0 { 0.0 } else { p })
                }
                (frequency, offset) => Phase {
                    frequency: Box::new(frequency),
                    offset: Box::new(offset),
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
        BinaryOp(BinaryOperator::Add, a, b) => {
            match (optimize(*a), optimize(*b)) {
                // Add yields the shorter of the two inputs.
                (Fixed(a, _), _) if a.is_empty() => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.is_empty() => Fixed(vec![], ()),
                (Const(a), Const(b)) => Const(BinaryOperator::Add.apply(a, b)),
                // Adding 0 is identity (because Add truncates to shorter, and Const is infinite)
                (a, Const(0.0)) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                // TODO I think re-associating the other way would mean a lower water mark for allocations (in
                // general). Consider long changes like in a big additive case.
                (BinaryOp(BinaryOperator::Add, a, b), Const(c)) => BinaryOp(
                    BinaryOperator::Add,
                    a,
                    Box::new(optimize(BinaryOp(
                        BinaryOperator::Add,
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
                    waveform: Box::new(optimize(BinaryOp(BinaryOperator::Add, a, b))),
                },
                (a, b) => BinaryOp(BinaryOperator::Add, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Subtract, a, b) => optimize(BinaryOp(
            BinaryOperator::Add,
            a,
            Box::new(optimize(BinaryOp(
                BinaryOperator::Multiply,
                b,
                Box::new(Const(-1.0)),
            ))),
        )),
        BinaryOp(BinaryOperator::Merge, a, b) => {
            use Waveform::Marked;
            match (optimize(*a), optimize(*b)) {
                // Merge yields the longer of the two inputs.
                (Fixed(a, _), b) if a.is_empty() => b,
                (a, Fixed(b, _)) if b.is_empty() => a,
                (Const(a), Const(b)) => Const(BinaryOperator::Merge.apply(a, b)),
                // Merging 0 is the identity if the left-hand side is infinite
                // TODO could check for other infinite waveforms
                (a @ (Time(_) | Noise), Const(0.0)) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryOp(
                    BinaryOperator::Merge,
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
                    Append(b, c, ()),
                ) => match *b {
                    Fin {
                        length: b_length,
                        waveform: b,
                    } if first_root(&a_length) == first_root(&b_length) => optimize(Append(
                        Box::new(Fin {
                            length: a_length,
                            waveform: Box::new(BinaryOp(BinaryOperator::Merge, a, b)),
                        }),
                        c,
                        (),
                    )),
                    _ => BinaryOp(
                        BinaryOperator::Merge,
                        Box::new(Fin {
                            length: a_length,
                            waveform: a,
                        }),
                        Box::new(Append(b, c, ())),
                    ),
                },
                // Same as above, but when w is Marked(n, w'). There's a lot of overlap
                // with the previous case!
                (Marked { id, waveform: a }, Append(b, c, ())) => match (*a, *b) {
                    (
                        Fin {
                            length: a_length,
                            waveform: a,
                        },
                        Fin {
                            length: b_length,
                            waveform: b,
                        },
                    ) if first_root(&a_length) == first_root(&b_length) => optimize(Append(
                        Box::new(Marked {
                            id,
                            waveform: Box::new(Fin {
                                length: a_length,
                                waveform: Box::new(BinaryOp(BinaryOperator::Merge, a, b)),
                            }),
                        }),
                        c,
                        (),
                    )),
                    (a, b) => BinaryOp(
                        BinaryOperator::Merge,
                        Box::new(Marked {
                            id,
                            waveform: Box::new(a),
                        }),
                        Box::new(Append(Box::new(b), c, ())),
                    ),
                },
                (a, b) => BinaryOp(BinaryOperator::Merge, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Multiply, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (Fixed(a, _), _) if a.is_empty() => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.is_empty() => Fixed(vec![], ()),
                (a, Const(1.0)) => a,
                // If a is infinite, then we can replace multiplication of zero with zero
                (Time(_) | Noise, Const(0.0)) => Const(0.0),
                (Const(a), Const(b)) => Const(BinaryOperator::Multiply.apply(a, b)),
                (Fixed(a, _), Const(b)) => Fixed(
                    a.into_iter()
                        .map(|x| BinaryOperator::Multiply.apply(x, b))
                        .collect(),
                    (),
                ),
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                (BinaryOp(BinaryOperator::Multiply, a, b), Const(c)) => BinaryOp(
                    BinaryOperator::Multiply,
                    a,
                    Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // Distribute
                // (a + b) * c == (a * c) + (b * c)
                (BinaryOp(BinaryOperator::Add, a, b), Const(c)) => BinaryOp(
                    BinaryOperator::Add,
                    Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        a,
                        Box::new(Const(c)),
                    ))),
                    Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // (a / b) * c == (a * c) / b
                (BinaryOp(BinaryOperator::Divide, a, b), Const(c)) => BinaryOp(
                    BinaryOperator::Divide,
                    Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
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
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryOp(BinaryOperator::Multiply, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Divide, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (_, Fixed(b, _)) if b.is_empty() => Fixed(vec![], ()),
                (Const(a), Const(b)) => Const(BinaryOperator::Divide.apply(a, b)),
                // Prefer multiplication by the reciprocal (zero when dividing
                // by zero, matching `BinaryOperator::Divide`).
                (a, Const(b)) => optimize(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(a),
                    Box::new(Const(BinaryOperator::Divide.apply(1.0, b))),
                )),
                // ((a / b) / c) == (a / (b * c))
                (BinaryOp(BinaryOperator::Divide, a, b), c) => BinaryOp(
                    BinaryOperator::Divide,
                    a,
                    Box::new(optimize(BinaryOp(BinaryOperator::Multiply, b, Box::new(c)))),
                ),
                // (a / (b / c)) == (a * c) / b
                (a, BinaryOp(BinaryOperator::Divide, b, c)) => BinaryOp(
                    BinaryOperator::Divide,
                    Box::new(optimize(BinaryOp(BinaryOperator::Multiply, Box::new(a), c))),
                    b,
                ),

                // Pull Fin out
                (Fin { length, waveform }, b) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Divide,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Divide,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryOp(BinaryOperator::Divide, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Power, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (Fixed(a, _), _) if a.is_empty() => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.is_empty() => Fixed(vec![], ()),
                // An infinite waveform to the zeroth power is the constant 1
                (Time(_) | Noise, Const(0.0)) => Const(1.0),
                (a, Const(1.0)) => a,
                (Const(a), Const(b)) => Const(BinaryOperator::Power.apply(a, b)),
                (Fixed(a, _), Const(b)) => Fixed(
                    a.into_iter()
                        .map(|x| BinaryOperator::Power.apply(x, b))
                        .collect(),
                    (),
                ),
                (a, b) => BinaryOp(BinaryOperator::Power, Box::new(a), Box::new(b)),
            }
        }
        UnaryOp(op, a) => match optimize(*a) {
            Fixed(mut a, _) => Fixed(a.iter_mut().map(|x| op.apply(*x)).collect(), ()),
            Const(a) => Const(op.apply(a)),
            a => UnaryOp(op, Box::new(a)),
        },
        // TODO if the waveform has constants, they can be pulled out; also
        // nested resets with the same trigger. Consider a naive version of the
        // triangle wave as an example
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
        } => match (
            optimize(*trigger),
            optimize(*positive_waveform),
            optimize(*negative_waveform),
        ) {
            (Const(v), positive_waveform, _) if v >= 0.0 => positive_waveform,
            (Const(v), _, negative_waveform) if v < 0.0 => negative_waveform,
            (trigger, positive_waveform, negative_waveform) => Alt {
                trigger: Box::new(trigger),
                positive_waveform: Box::new(positive_waveform),
                negative_waveform: Box::new(negative_waveform),
            },
        },
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
    fn fin_of_an_empty_waveform_is_empty() {
        let fin: Waveform<(), ()> = Fin {
            length: Box::new(BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Time(())),
                Box::new(Const(1.0)),
            )),
            waveform: Box::new(Fixed(vec![], ())),
        };
        assert_eq!(optimize(fin), Fixed(vec![], ()));
    }

    #[test]
    fn power_of_zero_folds_only_for_infinite_bases() {
        let infinite: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Power,
            Box::new(Time(())),
            Box::new(Const(0.0)),
        );
        assert_eq!(optimize(infinite), Const(1.0));
        let finite: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Power,
            Box::new(Fixed(vec![2.0, 3.0], ())),
            Box::new(Const(0.0)),
        );
        assert_eq!(optimize(finite), Fixed(vec![1.0, 1.0], ()));
    }

    #[test]
    fn test_optimize() {
        let w1: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Add,
            Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(Const(1.0)),
                Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Const(2.0)),
                    Box::new(Const(3.0)),
                )),
            )),
            Box::new(Const(4.0)),
        );
        assert_eq!(optimize(w1), Const(10.0));

        let w2: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Add,
            Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Const(3.0)),
                    Box::new(Time(())),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w2),
            BinaryOp(
                BinaryOperator::Add,
                Box::new(Time(())),
                Box::new(Const(10.0))
            ),
        );

        let w3: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Multiply,
            Box::new(BinaryOp(
                BinaryOperator::Multiply,
                Box::new(Const(2.0)),
                Box::new(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Time(())),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w3),
            BinaryOp(
                BinaryOperator::Multiply,
                Box::new(Time(())),
                Box::new(Const(30.0))
            ),
        );

        let w4: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Multiply,
            Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Time(())),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            optimize(w4),
            BinaryOp(
                BinaryOperator::Add,
                Box::new(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(Time(())),
                    Box::new(Const(15.0))
                )),
                Box::new(Const(10.0))
            ),
        );

        let w5: Waveform<(), ()> = BinaryOp(
            BinaryOperator::Multiply,
            Box::new(Fin {
                length: Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-2.0)),
                )),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(Fin {
                length: Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-1.5)),
                )),
                waveform: Box::new(Const(5.0)),
            }),
        );
        assert_eq!(
            optimize(w5),
            Fin {
                length: Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-1.5))
                )),
                waveform: Box::new(Const(15.0)),
            }
        );
    }
}
