use crate::tracker::Waveform;

// First root returns the first non-negative value at which the given waveform is zero. This is implemented for waveforms of the form Sum(Time, _), Time, and Const(0); returns None otherwise.
fn first_root(waveform: &Waveform) -> Option<Waveform> {
    use Waveform::*;
    match waveform {
        Const(0.0) => Some(Const(0.0)),
        Const(_) => None,
        Time => Some(Const(0.0)),
        Sum(a, b) => match (&**a, &**b) {
            // TODO should really check that Time doesn't appear on the other side too
            (Time, w) => Some(simplify(DotProduct(
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            (w, Time) => Some(simplify(DotProduct(
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            _ => None,
        },
        _ => None,
    }
}

// Assume offset waveforms are of the form `Time ~+ w` or `Const(x)`.
fn add_offsets(a: Waveform, b: Waveform) -> Waveform {
    use Waveform::*;
    match (first_root(&a), first_root(&b)) {
        (Some(a_root), Some(b_root)) => {
            let b = simplify(DotProduct(
                Box::new(Sum(Box::new(a_root), Box::new(b_root))),
                Box::new(Const(-1.0)),
            ));
            Sum(Box::new(Time), Box::new(b))
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
        w @ (Const(_) | Time | Noise | Fixed(_)) => w,
        Fin { length, waveform } => {
            let length = simplify(*length);
            match length {
                // Zero length
                Const(a) if a >= 0.0 => Fixed(vec![]),
                Fixed(v) if v.len() >= 1 && v[0] >= 0.0 => Fixed(vec![]),
                Time => Fixed(vec![]),
                length => match simplify(*waveform) {
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                    } => match (first_root(&length), first_root(&*inner_length)) {
                        (Some(Const(a)), Some(Const(b))) => Fin {
                            length: Box::new(Sum(Box::new(Time), Box::new(Const(-(a.min(b)))))),
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
        Append(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (a, b) {
                (Fixed(a), b) if a.len() == 0 => b,
                (a, Fixed(b)) if b.len() == 0 => a,
                (Fixed(a), Fixed(b)) => Fixed([a, b].concat()),
                (a, b) => Append(Box::new(a), Box::new(b)),
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
                waveform => Sin(Box::new(waveform)),
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
        Sum(a, b) => {
            match (simplify(*a), simplify(*b)) {
                (Fixed(a), b) if a.len() == 0 => b,
                (a, Fixed(b)) if b.len() == 0 => a,
                // NB. Can't collapse sums of Const(0.0) with finite waveforms as that would
                // change the length of the output.
                (Noise, Const(0.0)) => Noise,
                (Time, Const(0.0)) => Time,
                (Const(a), Const(b)) => Const(a + b),
                // Commute
                (Const(a), b) => simplify(Sum(Box::new(b), Box::new(Const(a)))),
                // Re-associate
                (Sum(a, b), Const(c)) => Sum(a, Box::new(simplify(Sum(b, Box::new(Const(c)))))),
                // TODO could distribute constants over Append(Fin, _), Res, and Alt
                // ... though currently Alt generates both branches, so better not to do too much work

                // Combine sum of Fin and an Append who first argument is Fin -- this occurs for expressions of
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
                    } if first_root(&a_length) == first_root(&b_length) => simplify(Append(
                        Box::new(Fin {
                            length: a_length,
                            waveform: Box::new(Sum(a, b)),
                        }),
                        c,
                    )),
                    _ => Sum(
                        Box::new(Fin {
                            length: a_length,
                            waveform: a,
                        }),
                        Box::new(Append(b, c)),
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
                    waveform: Box::new(Sum(a, b)),
                },
                (a, b) => Sum(Box::new(a), Box::new(b)),
            }
        }
        DotProduct(a, b) => {
            match (simplify(*a), simplify(*b)) {
                (Fixed(a), _) if a.len() == 0 => Fixed(vec![]),
                (_, Fixed(b)) if b.len() == 0 => Fixed(vec![]),
                (Const(0.0), _) => Fixed(vec![]),
                (_, Const(0.0)) => Fixed(vec![]),
                (Const(1.0), b) => b,
                (a, Const(1.0)) => a,
                (Const(a), Const(b)) => Const(a * b),
                // Commute
                (Const(a), b) => simplify(DotProduct(Box::new(b), Box::new(Const(a)))),
                // Re-associate
                (DotProduct(a, b), Const(c)) => {
                    DotProduct(a, Box::new(simplify(DotProduct(b, Box::new(Const(c))))))
                }
                // Distribute
                (Sum(a, b), Const(c)) => Sum(
                    Box::new(simplify(DotProduct(a, Box::new(Const(c))))),
                    Box::new(simplify(DotProduct(b, Box::new(Const(c))))),
                ),
                // TODO could distribute constants over, Append, Res, and Alt
                // ... though currently Alt generates both branches, so better not to do too much work

                // Pull Fin out
                (Fin { length, waveform }, b) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(DotProduct(waveform, Box::new(b)))),
                }),
                (a, Fin { length, waveform }) => simplify(Fin {
                    length,
                    waveform: Box::new(simplify(DotProduct(Box::new(a), waveform))),
                }),
                (a, b) => DotProduct(Box::new(a), Box::new(b)),
            }
        }
        Res { trigger, waveform } => Res {
            trigger: Box::new(simplify(*trigger)),
            waveform: Box::new(simplify(*waveform)),
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
            Sum(Box::new(Sin(Box::new(Time))), Box::new(Const(10.0))),
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
            DotProduct(Box::new(Sin(Box::new(Time))), Box::new(Const(30.0))),
        );

        let w4 = DotProduct(
            Box::new(Sum(
                Box::new(Const(2.0)),
                Box::new(DotProduct(
                    Box::new(Const(3.0)),
                    Box::new(Sin(Box::new(Time))),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w4),
            Sum(
                Box::new(DotProduct(
                    Box::new(Sin(Box::new(Time))),
                    Box::new(Const(15.0))
                )),
                Box::new(Const(10.0))
            ),
        );

        let w5 = DotProduct(
            Box::new(Fin {
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-2.0)))),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(Fin {
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-1.5)))),
                waveform: Box::new(Const(5.0)),
            }),
        );
        assert_eq!(
            simplify(w5),
            Fin {
                length: Box::new(Sum(Box::new(Time), Box::new(Const(-1.5)))),
                waveform: Box::new(Const(15.0)),
            }
        );
    }
}
