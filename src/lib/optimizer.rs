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

/// Returns the time at which `length` first becomes non-negative, if that time
/// is a constant.
fn constant_root<M>(length: &Waveform<M>) -> Option<f32>
where
    M: Clone + PartialEq,
{
    match first_root(length)? {
        Waveform::Const(root) => Some(root),
        _ => None,
    }
}

/// Optimizes a waveform through algebraic transformations which preserve its
/// length and its output, up to small differences in sample values and some
/// changes to floating-point special values.
///
/// Some examples include:
///   * eliminating constants
///   * pulling Fin's up and combining nested Fin's
///   * replacing zero-length waveforms with the canonical `Fixed(vec![])`
///   * handling some common cases of Fin in Add and Multiply
///
/// In addition, `optimize` also anticipates the implementation of
/// `Generator::generate` and re-associates binary operations so that Const
/// waveforms are on the right and Fixed waveforms are on the left.
///
/// Throughout this, `optimize` preserves any Marked or Captured waveforms.
pub fn optimize<M>(waveform: Waveform<M>) -> Waveform<M>
where
    M: Clone + PartialEq,
{
    use Waveform::*;
    match waveform {
        // No changes for these:
        w @ (Const(_) | Time(_) | Noise | Fixed(_, _)) => w,
        Fin {
            length, waveform, ..
        } => {
            let length = optimize(*length);
            match length {
                // Zero length
                Const(a) if a >= 0.0 && !has_marked_or_captured(&waveform) => Fixed(vec![], ()),
                Fixed(v, _)
                    if !v.is_empty() && v[0] >= 0.0 && !has_marked_or_captured(&waveform) =>
                {
                    Fixed(vec![], ())
                }
                // TODO for longer Fixed, replace with * Fixed(vec![1.0; v.len()])?
                Time(_) if !has_marked_or_captured(&waveform) => Fixed(vec![], ()),
                length => match optimize(*waveform) {
                    // Truncating an empty waveform leaves it empty.
                    Fixed(v, _) if v.is_empty() && !has_marked_or_captured(&length) => {
                        Fixed(vec![], ())
                    }
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                        ..
                    } => match (constant_root(&length), constant_root(&inner_length)) {
                        (Some(a), Some(b)) => Fin {
                            length: Box::new(optimize(BinaryOp(
                                BinaryOperator::Subtract,
                                Box::new(Time(())),
                                Box::new(Const(a.min(b))),
                            ))),
                            waveform,
                            state: (),
                        },
                        _ => Fin {
                            length: Box::new(length),
                            waveform: Box::new(Fin {
                                length: inner_length,
                                waveform,
                                state: (),
                            }),
                            state: (),
                        },
                    },
                    waveform => Fin {
                        length: Box::new(length),
                        waveform: Box::new(waveform),
                        state: (),
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
        } => {
            let waveform = optimize(*waveform);
            let feed_forward: Vec<_> = feed_forward.into_iter().map(optimize).collect();
            let feedback = feedback.into_iter().map(optimize).collect();
            match (waveform, feed_forward.as_slice()) {
                // A single non-constant feed-forward coefficient only scales
                // the current input sample, so apply it to the input instead,
                // leaving a constant coefficient. The coefficient is
                // zero-extended, so the product must be too, which keeps the
                // input's length only if the input is infinite.
                (waveform, [coefficient])
                    if !is_possibly_finite(&waveform) && !matches!(coefficient, Const(_)) =>
                {
                    Filter {
                        waveform: Box::new(optimize(BinaryOp(
                            BinaryOperator::Merge,
                            Box::new(BinaryOp(
                                BinaryOperator::Multiply,
                                Box::new(coefficient.clone()),
                                Box::new(waveform),
                            )),
                            Box::new(Const(0.0)),
                        ))),
                        feed_forward: vec![Const(1.0)],
                        feedback,
                        state,
                    }
                }
                (waveform, _) => Filter {
                    waveform: Box::new(waveform),
                    feed_forward,
                    feedback,
                    state,
                },
            }
        }
        BinaryOp(BinaryOperator::Add, a, b) => {
            match (optimize(*a), optimize(*b)) {
                // Add yields the shorter of the two inputs.
                (Fixed(a, _), b) if a.is_empty() && !has_marked_or_captured(&b) => {
                    Fixed(vec![], ())
                }
                (a, Fixed(b, _)) if b.is_empty() && !has_marked_or_captured(&a) => {
                    Fixed(vec![], ())
                }
                (Const(a), Const(b)) => Const(BinaryOperator::Add.apply(a, b)),
                // Adding 0 is identity (because Add truncates to shorter, and Const is infinite)
                (a, Const(0.0)) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Commute (moving Fixed to the left)
                (a, Fixed(b, _)) if !matches!(a, Fixed(..)) => optimize(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Fixed(b, ())),
                    Box::new(a),
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
                        ..
                    },
                    Fin {
                        length: b_length,
                        waveform: b,
                        ..
                    },
                ) if constant_root(&a_length)
                    .is_some_and(|a| Some(a) == constant_root(&b_length)) =>
                {
                    Fin {
                        length: a_length,
                        waveform: Box::new(optimize(BinaryOp(BinaryOperator::Add, a, b))),
                        state: (),
                    }
                }
                (
                    a,
                    Fin {
                        length, waveform, ..
                    },
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Add,
                        Box::new(a),
                        waveform,
                    ))),
                    state: (),
                }),
                (
                    Fin {
                        length, waveform, ..
                    },
                    b,
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Add,
                        waveform,
                        Box::new(b),
                    ))),
                    state: (),
                }),
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
            match (optimize(*a), optimize(*b)) {
                // Merge yields the longer of the two inputs.
                (Fixed(a, _), b) if a.is_empty() => b,
                (a, Fixed(b, _)) if b.is_empty() => a,
                (Const(a), Const(b)) => Const(BinaryOperator::Merge.apply(a, b)),
                // Merging 0 is the identity if the left-hand side is infinite
                (a, Const(0.0)) if !is_possibly_finite(&a) => a,
                // Commute (moving constants to the right)
                (Const(a), b) => optimize(BinaryOp(
                    BinaryOperator::Merge,
                    Box::new(b),
                    Box::new(Const(a)),
                )),
                // Commute (moving Fixed to the left)
                (a, Fixed(b, _)) if !matches!(a, Fixed(..)) => optimize(BinaryOp(
                    BinaryOperator::Merge,
                    Box::new(Fixed(b, ())),
                    Box::new(a),
                )),
                // Combine merge of Fin and an Append who first argument is Fin -- this occurs for expressions of
                // the form `w | fin(t) | seq(t)`.
                (
                    Fin {
                        length: a_length,
                        waveform: a,
                        ..
                    },
                    Append(b, c, ()),
                ) => match *b {
                    Fin {
                        length: b_length,
                        waveform: b,
                        ..
                    } if constant_root(&a_length)
                        .is_some_and(|a| Some(a) == constant_root(&b_length)) =>
                    {
                        optimize(Append(
                            Box::new(Fin {
                                length: a_length,
                                waveform: Box::new(BinaryOp(BinaryOperator::Merge, a, b)),
                                state: (),
                            }),
                            c,
                            (),
                        ))
                    }
                    _ => BinaryOp(
                        BinaryOperator::Merge,
                        Box::new(Fin {
                            length: a_length,
                            waveform: a,
                            state: (),
                        }),
                        Box::new(Append(b, c, ())),
                    ),
                },
                (a, b) => BinaryOp(BinaryOperator::Merge, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Multiply, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (Fixed(a, _), b) if a.is_empty() && !has_marked_or_captured(&b) => {
                    Fixed(vec![], ())
                }
                (a, Fixed(b, _)) if b.is_empty() && !has_marked_or_captured(&a) => {
                    Fixed(vec![], ())
                }
                (a, Const(1.0)) => a,
                // If a is infinite, then we can replace multiplication of zero with zero
                (a, Const(0.0)) if !is_possibly_finite(&a) && !has_marked_or_captured(&a) => {
                    Const(0.0)
                }
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
                // Commute (moving Fixed to the left)
                (a, Fixed(b, _)) if !matches!(a, Fixed(..)) => optimize(BinaryOp(
                    BinaryOperator::Multiply,
                    Box::new(Fixed(b, ())),
                    Box::new(a),
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
                (
                    Fin {
                        length, waveform, ..
                    },
                    b,
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        waveform,
                        Box::new(b),
                    ))),
                    state: (),
                }),
                (
                    a,
                    Fin {
                        length, waveform, ..
                    },
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Multiply,
                        Box::new(a),
                        waveform,
                    ))),
                    state: (),
                }),
                (a, b) => BinaryOp(BinaryOperator::Multiply, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Divide, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (a, Fixed(b, _)) if b.is_empty() && !has_marked_or_captured(&a) => {
                    Fixed(vec![], ())
                }
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
                (
                    Fin {
                        length, waveform, ..
                    },
                    b,
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Divide,
                        waveform,
                        Box::new(b),
                    ))),
                    state: (),
                }),
                (
                    a,
                    Fin {
                        length, waveform, ..
                    },
                ) => optimize(Fin {
                    length,
                    waveform: Box::new(optimize(BinaryOp(
                        BinaryOperator::Divide,
                        Box::new(a),
                        waveform,
                    ))),
                    state: (),
                }),
                (a, b) => BinaryOp(BinaryOperator::Divide, Box::new(a), Box::new(b)),
            }
        }
        BinaryOp(BinaryOperator::Power, a, b) => {
            match (optimize(*a), optimize(*b)) {
                (Fixed(a, _), b) if a.is_empty() && !has_marked_or_captured(&b) => {
                    Fixed(vec![], ())
                }
                (a, Fixed(b, _)) if b.is_empty() && !has_marked_or_captured(&a) => {
                    Fixed(vec![], ())
                }
                // An infinite waveform to the zeroth power is the constant 1
                (a, Const(0.0)) if !is_possibly_finite(&a) && !has_marked_or_captured(&a) => {
                    Const(1.0)
                }
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
        BinaryOp(BinaryOperator::Log, a, b) => match (optimize(*a), optimize(*b)) {
            (Fixed(a, _), b) if a.is_empty() && !has_marked_or_captured(&b) => Fixed(vec![], ()),
            (a, Fixed(b, _)) if b.is_empty() && !has_marked_or_captured(&a) => Fixed(vec![], ()),
            // Constant folding.
            (Const(a), Const(b)) => Const(BinaryOperator::Log.apply(a, b)),
            // Fixed with a constant base: map element-wise.
            (Fixed(a, _), Const(b)) => Fixed(
                a.into_iter()
                    .map(|x| BinaryOperator::Log.apply(x, b))
                    .collect(),
                (),
            ),
            // Constant argument with a Fixed base: also element-wise.
            (Const(a), Fixed(b, _)) => Fixed(
                b.into_iter()
                    .map(|y| BinaryOperator::Log.apply(a, y))
                    .collect(),
                (),
            ),
            (a, b) => BinaryOp(BinaryOperator::Log, Box::new(a), Box::new(b)),
        },
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
            (Const(v), positive_waveform, negative_waveform)
                if v >= 0.0 && !has_marked_or_captured(&negative_waveform) =>
            {
                positive_waveform
            }
            (Const(v), positive_waveform, negative_waveform)
                if v < 0.0 && !has_marked_or_captured(&positive_waveform) =>
            {
                negative_waveform
            }
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

/// Returns false only if waveform is certainly infinite.
///
/// Marked waveforms count as possibly finite because their contents can change,
/// as do Captured waveforms so the same samples are always captured.
fn is_possibly_finite<M, S>(waveform: &Waveform<M, S>) -> bool {
    use Waveform::*;
    match waveform {
        Const(_) | Time(_) | Noise => false,
        Fixed(_, _) => true,
        Fin { .. } => true,
        Append(a, b, _) => is_possibly_finite(a) && is_possibly_finite(b),
        Phase {
            frequency, offset, ..
        } => is_possibly_finite(frequency) || is_possibly_finite(offset),
        Filter { waveform, .. } => is_possibly_finite(waveform),
        BinaryOp(op, a, b) => {
            if op.extends_to_longer() {
                is_possibly_finite(a) && is_possibly_finite(b)
            } else {
                is_possibly_finite(a) || is_possibly_finite(b)
            }
        }
        UnaryOp(_, a) => is_possibly_finite(a),
        Reset { trigger, .. } | Alt { trigger, .. } => is_possibly_finite(trigger),
        Marked { .. } | Captured { .. } => true,
    }
}

/// Returns true if `waveform` contains a `Marked` or `Captured` waveform.
fn has_marked_or_captured<M, S>(waveform: &Waveform<M, S>) -> bool {
    use Waveform::*;
    match waveform {
        Const(_) | Time(_) | Noise | Fixed(_, _) => false,
        Marked { .. } | Captured { .. } => true,
        UnaryOp(_, a) => has_marked_or_captured(a),
        Fin {
            length: a,
            waveform: b,
            ..
        }
        | Append(a, b, _)
        | Phase {
            frequency: a,
            offset: b,
            ..
        }
        | BinaryOp(_, a, b)
        | Reset {
            trigger: a,
            waveform: b,
            ..
        } => has_marked_or_captured(a) || has_marked_or_captured(b),
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => {
            has_marked_or_captured(waveform)
                || feed_forward
                    .iter()
                    .chain(feedback)
                    .any(has_marked_or_captured)
        }
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            has_marked_or_captured(trigger)
                || has_marked_or_captured(positive_waveform)
                || has_marked_or_captured(negative_waveform)
        }
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
            state: (),
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
    fn single_feed_forward_coefficient_moves_to_infinite_input() {
        let filter = |waveform: Waveform<(), ()>, feed_forward: Vec<Waveform<(), ()>>| Filter {
            waveform: Box::new(waveform),
            feed_forward,
            feedback: vec![Const(-1.0)],
            state: (),
        };
        let hold = Fixed(vec![1.0], ());
        assert_eq!(
            optimize(filter(Noise, vec![hold.clone()])),
            filter(
                BinaryOp(
                    BinaryOperator::Merge,
                    Box::new(BinaryOp(
                        BinaryOperator::Multiply,
                        Box::new(hold.clone()),
                        Box::new(Noise),
                    )),
                    Box::new(Const(0.0)),
                ),
                vec![Const(1.0)],
            )
        );
        // A finite input would be extended by the zero-extended product.
        let finite = filter(Fixed(vec![2.0, 3.0], ()), vec![hold.clone()]);
        assert_eq!(optimize(finite.clone()), finite);
        // The first coefficient doesn't scale the delayed samples read by the others.
        let two = filter(Noise, vec![hold.clone(), Const(0.5)]);
        assert_eq!(optimize(two.clone()), two);
    }

    #[test]
    fn adding_a_constant_to_fin_moves_inside_it() {
        let fin: Waveform<(), ()> = Fin {
            length: Box::new(BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Time(())),
                Box::new(Const(1.0)),
            )),
            waveform: Box::new(Time(())),
            state: (),
        };
        let expected = Fin {
            length: Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(Time(())),
                Box::new(Const(-1.0)),
            )),
            waveform: Box::new(BinaryOp(
                BinaryOperator::Add,
                Box::new(Time(())),
                Box::new(Const(0.5)),
            )),
            state: (),
        };
        let fin_plus_const = BinaryOp(
            BinaryOperator::Add,
            Box::new(fin.clone()),
            Box::new(Const(0.5)),
        );
        assert_eq!(optimize(fin_plus_const), expected);
        let const_plus_fin = BinaryOp(BinaryOperator::Add, Box::new(Const(0.5)), Box::new(fin));
        assert_eq!(optimize(const_plus_fin), expected);
    }

    #[test]
    fn marked_waveforms_are_never_removed() {
        let mark = || Marked {
            id: (),
            waveform: Box::new(Const(0.3)),
        };
        let op = |op, a, b| BinaryOp(op, Box::new(a), Box::new(b));
        let fin = |length, waveform| Fin {
            length: Box::new(length),
            waveform: Box::new(waveform),
            state: (),
        };
        let empty = || Fixed(vec![], ());
        let infinite_with_mark = || op(BinaryOperator::Merge, mark(), Time(()));
        // A length without a constant root.
        let unsolved = || op(BinaryOperator::Multiply, Time(()), Time(()));
        let mut cases = vec![
            fin(Const(0.0), mark()),
            fin(Fixed(vec![1.0], ()), mark()),
            fin(Time(()), mark()),
            fin(mark(), empty()),
            op(BinaryOperator::Multiply, infinite_with_mark(), Const(0.0)),
            op(BinaryOperator::Power, infinite_with_mark(), Const(0.0)),
            op(BinaryOperator::Divide, infinite_with_mark(), Const(0.0)),
            op(BinaryOperator::Divide, mark(), empty()),
            op(
                BinaryOperator::Add,
                fin(unsolved(), Time(())),
                fin(mark(), Time(())),
            ),
            op(
                BinaryOperator::Merge,
                fin(unsolved(), Time(())),
                Append(Box::new(fin(mark(), Time(()))), Box::new(Time(())), ()),
            ),
            Alt {
                trigger: Box::new(Const(1.0)),
                positive_waveform: Box::new(Time(())),
                negative_waveform: Box::new(mark()),
            },
            Alt {
                trigger: Box::new(Const(-1.0)),
                positive_waveform: Box::new(mark()),
                negative_waveform: Box::new(Time(())),
            },
        ];
        for operator in [
            BinaryOperator::Add,
            BinaryOperator::Multiply,
            BinaryOperator::Power,
            BinaryOperator::Log,
        ] {
            cases.push(op(operator, empty(), mark()));
            cases.push(op(operator, mark(), empty()));
        }
        for w in cases {
            let optimized = optimize(w.clone());
            assert!(has_marked_or_captured(&optimized), "{w:?} => {optimized:?}");
        }
    }

    #[test]
    fn fins_combine_only_with_the_same_constant_root() {
        let op = |op, a, b| BinaryOp(op, Box::new(a), Box::new(b));
        let fin = |length, waveform| Fin {
            length: Box::new(length),
            waveform: Box::new(waveform),
            state: (),
        };
        let scaled_second = |k| {
            op(
                BinaryOperator::Add,
                op(BinaryOperator::Multiply, Time(()), Const(k)),
                Const(-1.0),
            )
        };
        let one_second = || op(BinaryOperator::Add, Time(()), Const(-1.0));
        let unequal: Waveform<(), ()> = op(
            BinaryOperator::Add,
            fin(scaled_second(2.0), Const(1.0)),
            fin(scaled_second(4.0), Const(2.0)),
        );
        assert_eq!(
            optimize(unequal),
            fin(scaled_second(4.0), fin(scaled_second(2.0), Const(3.0)))
        );
        let equal: Waveform<(), ()> = op(
            BinaryOperator::Add,
            fin(one_second(), Const(1.0)),
            fin(one_second(), Const(2.0)),
        );
        assert_eq!(optimize(equal), fin(one_second(), Const(3.0)));
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
                state: (),
            }),
            Box::new(Fin {
                length: Box::new(BinaryOp(
                    BinaryOperator::Add,
                    Box::new(Time(())),
                    Box::new(Const(-1.5)),
                )),
                waveform: Box::new(Const(5.0)),
                state: (),
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
                state: ()
            }
        );
    }
}
