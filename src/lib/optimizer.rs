use crate::waveform::{Operator, Waveform};

// First root returns the first non-negative value at which the given waveform is zero. This is implemented for waveforms of the form BinaryPointOp(Operator::Add|Operator::Subtract, Time, _), Time, and Const(0); returns None otherwise.
fn first_root(waveform: &Waveform) -> Option<Waveform> {
    use Waveform::*;
    match waveform {
        Const(0.0) => Some(Const(0.0)),
        Const(_) => None,
        Time(_) => Some(Const(0.0)),
        BinaryPointOp(Operator::Add, a, b) => match (&**a, &**b) {
            // TODO should really check that Time doesn't appear on the other side too
            (Time(_), w) => Some(simplify(BinaryPointOp(
                Operator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            (w, Time(_)) => Some(simplify(BinaryPointOp(
                Operator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            _ => None,
        },
        BinaryPointOp(Operator::Subtract, a, b) => first_root(&BinaryPointOp(
            Operator::Add,
            a.clone(),
            Box::new(simplify(BinaryPointOp(
                Operator::Multiply,
                b.clone(),
                Box::new(Const(-1.0)),
            ))),
        )),
        _ => None,
    }
}

// Replaces all Seq's with Appends and returns the overall offset of the waveform.
pub fn replace_seq(waveform: Waveform) -> (Waveform, Waveform) {
    use Waveform::*;

    // Assume offset waveforms are of the form `Time ~+ w` or `Const(x)`.
    fn add_offsets(a: Waveform, b: Waveform) -> Waveform {
        match (first_root(&a), first_root(&b)) {
            (Some(a_root), Some(b_root)) => {
                let b = simplify(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(BinaryPointOp(
                        Operator::Add,
                        Box::new(a_root),
                        Box::new(b_root),
                    )),
                    Box::new(Const(-1.0)),
                ));
                BinaryPointOp(Operator::Add, Box::new(Time(())), Box::new(b))
            }
            (a_root, b_root) => {
                panic!(
                    "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
                    a_root, b_root, a, b
                );
            }
        }
    }

    match waveform {
        w @ (Const(_) | Time(_) | Noise | Fixed(_, _)) => (Const(0.0), w),
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
        Append(a, b, state) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            (
                add_offsets(a_offset, b_offset),
                Append(Box::new(a), Box::new(b), state),
            )
        }
        Sin {
            frequency,
            phase,
            state,
        } => {
            let (a_offset, a) = replace_seq(*frequency);
            let (b_offset, b) = replace_seq(*phase);
            let offset = add_offsets(a_offset, b_offset);
            (
                offset,
                Sin {
                    frequency: Box::new(a),
                    phase: Box::new(b),
                    state,
                },
            )
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
        BinaryPointOp(op @ (Operator::Add | Operator::Subtract), a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            if a_offset == Const(0.0) {
                return (b_offset, BinaryPointOp(op, Box::new(a), Box::new(b)));
            }
            (
                add_offsets(a_offset.clone(), b_offset),
                BinaryPointOp(
                    op,
                    Box::new(a),
                    Box::new(Append(
                        Box::new(Fin {
                            length: Box::new(a_offset),
                            waveform: Box::new(Const(0.0)),
                        }),
                        Box::new(b),
                        (),
                    )),
                ),
            )
        }
        BinaryPointOp(op @ (Operator::Multiply | Operator::Divide), a, b) => {
            let (a_offset, a) = replace_seq(*a);
            let (b_offset, b) = replace_seq(*b);
            if a_offset == Const(0.0) {
                return (b_offset, BinaryPointOp(op, Box::new(a), Box::new(b)));
            }
            (
                add_offsets(a_offset.clone(), b_offset),
                BinaryPointOp(
                    op,
                    Box::new(a),
                    Box::new(Append(
                        Box::new(Fin {
                            length: Box::new(a_offset),
                            waveform: Box::new(Const(1.0)),
                        }),
                        Box::new(b),
                        (),
                    )),
                ),
            )
        }
        Res {
            trigger,
            waveform,
            state,
        } => {
            let (offset, trigger) = replace_seq(*trigger);
            (
                offset,
                Res {
                    trigger: Box::new(trigger),
                    waveform: Box::new(replace_seq(*waveform).1),
                    state,
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

// Simplify waveform expressions by...
//   * eliminating constants in binary operators and Sin
//   * re-associates binary operations so that Consts are on the right
//   * pulling Fin's up and combining nested Fin's
//   * replacing zero-length waveforms with the canonical `Fixed(vec![])`
//   * handling some common cases of Fin in Sum and DotProduct
// This function must be called after replace_seq.
pub fn simplify(waveform: Waveform) -> Waveform {
    use Waveform::*;
    match waveform {
        // No changes for these:
        w @ (Const(_) | Time(_) | Noise | Fixed(_, _)) => w,
        Fin { length, waveform } => {
            let length = simplify(*length);
            match length {
                // Zero length
                Const(a) if a >= 0.0 => Fixed(vec![], ()),
                Fixed(v, _) if v.len() >= 1 && v[0] >= 0.0 => Fixed(vec![], ()),
                // TODO for longer Fixed, replace with * Fixed(vec![1.0; v.len()])?
                Time(_) => Fixed(vec![], ()),
                length => match simplify(*waveform) {
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                    } => match (first_root(&length), first_root(&*inner_length)) {
                        (Some(Const(a)), Some(Const(b))) => Fin {
                            length: Box::new(simplify(BinaryPointOp(
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
        Seq { .. } => {
            panic!("Seq should have been replaced by replace_seq before simplify is called");
        }
        Append(a, b, _) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (a, b) {
                (Fixed(a, _), b) if a.len() == 0 => b,
                (a, Fixed(b, _)) if b.len() == 0 => a,
                (Fixed(a, _), Fixed(b, _)) => Fixed([a, b].concat(), ()),
                (a, b) => Append(Box::new(a), Box::new(b), ()),
            }
        }
        // Check to see if we can compute the sine function:
        Sin {
            frequency,
            phase,
            state,
        } => {
            let frequency = simplify(*frequency);
            let phase = simplify(*phase);
            match (frequency, phase) {
                (Const(0.0), Const(p)) => Const(p.sin()),
                (Const(0.0), Fixed(v, _)) => {
                    let v = v.into_iter().map(|x| x.sin()).collect();
                    Fixed(v, ())
                }
                (frequency, phase) => Sin {
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
            waveform: Box::new(simplify(*waveform)),
            feed_forward: Box::new(simplify(*feed_forward)),
            feedback: Box::new(simplify(*feedback)),
            state,
        },
        BinaryPointOp(Operator::Add, a, b) => {
            match (simplify(*a), simplify(*b)) {
                (Fixed(a, _), b) if a.len() == 0 => b,
                (a, Fixed(b, _)) if b.len() == 0 => a,
                // NB. Can't collapse sums of Const(0.0) with finite waveforms as that would
                // change the length of the output.
                (Noise, Const(0.0)) => Noise,
                (Time(_), Const(0.0)) => Time(()),
                (Const(a), Const(b)) => Const(a + b),
                (Slider(s), Const(0.0)) => Slider(s),
                // Commute (moving constants to the right)
                (Const(a), b) => simplify(BinaryPointOp(
                    Operator::Add,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                (BinaryPointOp(Operator::Add, a, b), Const(c)) => BinaryPointOp(
                    Operator::Add,
                    a,
                    Box::new(simplify(BinaryPointOp(
                        Operator::Add,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // TODO could distribute constants over Append(Fin, _), Res, and Alt
                // ... though currently Alt generates both branches, so better not to do too much work

                // Combine sum of Fin and an Append who first argument is Fin -- this occurs for expressions of
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
                    } if first_root(&a_length) == first_root(&b_length) => simplify(Append(
                        Box::new(Fin {
                            length: a_length,
                            waveform: Box::new(BinaryPointOp(Operator::Add, a, b)),
                        }),
                        c,
                        (),
                    )),
                    _ => BinaryPointOp(
                        Operator::Add,
                        Box::new(Fin {
                            length: a_length,
                            waveform: a,
                        }),
                        Box::new(Append(b, c, ())),
                    ),
                },
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
                    waveform: Box::new(simplify(BinaryPointOp(Operator::Add, a, b))),
                },
                (a, b) => BinaryPointOp(Operator::Add, Box::new(a), Box::new(b)),
            }
        }
        BinaryPointOp(Operator::Subtract, a, b) => simplify(BinaryPointOp(
            Operator::Add,
            a,
            Box::new(simplify(BinaryPointOp(
                Operator::Multiply,
                b,
                Box::new(Const(-1.0)),
            ))),
        )),
        BinaryPointOp(Operator::Multiply, a, b) => {
            match (simplify(*a), simplify(*b)) {
                (Fixed(a, _), _) if a.len() == 0 => Fixed(vec![], ()),
                (_, Fixed(b, _)) if b.len() == 0 => Fixed(vec![], ()),
                (Const(1.0), b) => b,
                (a, Const(1.0)) => a,
                (Const(a), Const(b)) => Const(a * b),
                (Fixed(a, _), Const(b)) => Fixed(a.into_iter().map(|x| x * b).collect(), ()),
                // Commute (moving constants to the right)
                (Const(a), b) => simplify(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Re-associate
                (BinaryPointOp(Operator::Multiply, a, b), Const(c)) => BinaryPointOp(
                    Operator::Multiply,
                    a,
                    Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // Distribute
                // (a + b) * c == (a * c) + (b * c)
                (BinaryPointOp(Operator::Add, a, b), Const(c)) => BinaryPointOp(
                    Operator::Add,
                    Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        a,
                        Box::new(Const(c)),
                    ))),
                    Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        b,
                        Box::new(Const(c)),
                    ))),
                ),
                // (a / b) * c == (a * c) / b
                (BinaryPointOp(Operator::Divide, a, b), Const(c)) => BinaryPointOp(
                    Operator::Divide,
                    Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        a,
                        Box::new(Const(c)),
                    ))),
                    b,
                ),
                // TODO could distribute constants over, Append, Res, and Alt
                // ... though currently Alt generates both branches, so better not to do too much work

                // Pull Fin out
                (Fin { length, waveform }, b) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(BinaryPointOp(
                        Operator::Multiply,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryPointOp(Operator::Multiply, Box::new(a), Box::new(b)),
            }
        }
        BinaryPointOp(Operator::Divide, a, b) => {
            match (simplify(*a), simplify(*b)) {
                (_, Fixed(b, _)) if b.len() == 0 => Fixed(vec![], ()),
                // Prefer multiplication
                (a, Const(b)) => simplify(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(a),
                    Box::new(Const(1.0 / b)),
                )),
                // ((a / b) / c) == (a / (b * c))
                (BinaryPointOp(Operator::Divide, a, b), c) => BinaryPointOp(
                    Operator::Divide,
                    a,
                    Box::new(simplify(BinaryPointOp(Operator::Multiply, b, Box::new(c)))),
                ),
                // (a / (b / c)) == (a * c) / b
                (a, BinaryPointOp(Operator::Divide, b, c)) => BinaryPointOp(
                    Operator::Divide,
                    Box::new(simplify(BinaryPointOp(Operator::Multiply, Box::new(a), c))),
                    b,
                ),

                // Pull Fin out
                (Fin { length, waveform }, b) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(BinaryPointOp(
                        Operator::Divide,
                        waveform,
                        Box::new(b),
                    ))),
                }),
                (a, Fin { length, waveform }) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(BinaryPointOp(
                        Operator::Divide,
                        Box::new(a),
                        waveform,
                    ))),
                }),
                (a, b) => BinaryPointOp(Operator::Divide, Box::new(a), Box::new(b)),
            }
        }
        Res {
            trigger,
            waveform,
            state,
        } => Res {
            trigger: Box::new(simplify(*trigger)),
            waveform: Box::new(simplify(*waveform)),
            state,
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: Box::new(simplify(*trigger)),
            positive_waveform: Box::new(simplify(*positive_waveform)),
            negative_waveform: Box::new(simplify(*negative_waveform)),
        },
        w @ Slider(_) => w,
        Marked { id, waveform } => {
            // TODO could pull out Fin if process_marks better implemented Fin
            Marked {
                id,
                waveform: Box::new(simplify(*waveform)),
            }
        }
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: Box::new(simplify(*waveform)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Waveform::*;

    #[test]
    fn test_simplify() {
        let w1 = BinaryPointOp(
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
        assert_eq!(simplify(w1), Const(10.0));

        let w2 = BinaryPointOp(
            Operator::Add,
            Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Const(3.0)),
                    Box::new(Sin {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w2),
            BinaryPointOp(
                Operator::Add,
                Box::new(Sin {
                    frequency: Box::new(Const(1.0)),
                    phase: Box::new(Const(0.0)),
                    state: ()
                }),
                Box::new(Const(10.0))
            ),
        );

        let w3 = BinaryPointOp(
            Operator::Multiply,
            Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Sin {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w3),
            BinaryPointOp(
                Operator::Multiply,
                Box::new(Sin {
                    frequency: Box::new(Const(1.0)),
                    phase: Box::new(Const(0.0)),
                    state: ()
                }),
                Box::new(Const(30.0))
            ),
        );

        let w4 = BinaryPointOp(
            Operator::Multiply,
            Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(Const(2.0)),
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Const(3.0)),
                    Box::new(Sin {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: (),
                    }),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w4),
            BinaryPointOp(
                Operator::Add,
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Sin {
                        frequency: Box::new(Const(1.0)),
                        phase: Box::new(Const(0.0)),
                        state: ()
                    }),
                    Box::new(Const(15.0))
                )),
                Box::new(Const(10.0))
            ),
        );

        let w5 = BinaryPointOp(
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
            simplify(w5),
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
