use crate::tracker::Generator;
use crate::tracker::Operator;
use crate::tracker::Waveform;

// First root returns the first non-negative value at which the given waveform is zero. This is implemented for waveforms of the form BinaryPointOp(Operator::Add|Operator::Subtract, Time, _), Time, and Const(0); returns None otherwise.
fn first_root(waveform: &Waveform) -> Option<Waveform> {
    use Waveform::*;
    match waveform {
        Const(0.0) => Some(Const(0.0)),
        Const(_) => None,
        Time => Some(Const(0.0)),
        BinaryPointOp(Operator::Add, a, b) => match (&**a, &**b) {
            // TODO should really check that Time doesn't appear on the other side too
            (Time, w) => Some(simplify(BinaryPointOp(
                Operator::Multiply,
                Box::new(w.clone()),
                Box::new(Const(-1.0)),
            ))),
            (w, Time) => Some(simplify(BinaryPointOp(
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
                BinaryPointOp(Operator::Add, Box::new(Time), Box::new(b))
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
        Sin(waveform, state) => {
            let (offset, waveform) = replace_seq(*waveform);
            (offset, Sin(Box::new(waveform), state))
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
        w @ (Const(_) | Time | Noise | Fixed(_)) => w,
        Fin { length, waveform } => {
            let length = simplify(*length);
            match length {
                // Zero length
                Const(a) if a >= 0.0 => Fixed(vec![]),
                Fixed(v) if v.len() >= 1 && v[0] >= 0.0 => Fixed(vec![]),
                // TODO for longer Fixed, replace with * Fixed(vec![1.0; v.len()])?
                Time => Fixed(vec![]),
                length => match simplify(*waveform) {
                    // Nested Fin's
                    Fin {
                        length: inner_length,
                        waveform,
                    } => match (first_root(&length), first_root(&*inner_length)) {
                        (Some(Const(a)), Some(Const(b))) => Fin {
                            length: Box::new(simplify(BinaryPointOp(
                                Operator::Subtract,
                                Box::new(Time),
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
        // Check to see if we can compute the sine function:
        Sin(waveform, state) => {
            let waveform = simplify(*waveform);
            match waveform {
                Const(a) => Const(a.sin()),
                Fixed(v) => {
                    let v = v.into_iter().map(|x| x.sin()).collect();
                    Fixed(v)
                }
                waveform => Sin(Box::new(waveform), state),
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
                (Fixed(a), b) if a.len() == 0 => b,
                (a, Fixed(b)) if b.len() == 0 => a,
                // NB. Can't collapse sums of Const(0.0) with finite waveforms as that would
                // change the length of the output.
                (Noise, Const(0.0)) => Noise,
                (Time, Const(0.0)) => Time,
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
                    Append(b, c),
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
                    )),
                    _ => BinaryPointOp(
                        Operator::Add,
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
                (Fixed(a), _) if a.len() == 0 => Fixed(vec![]),
                (_, Fixed(b)) if b.len() == 0 => Fixed(vec![]),
                (Const(1.0), b) => b,
                (a, Const(1.0)) => a,
                (Const(a), Const(b)) => Const(a * b),
                (Fixed(a), Const(b)) => Fixed(a.into_iter().map(|x| x * b).collect()),
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
                (_, Fixed(b)) if b.len() == 0 => Fixed(vec![]),
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

use crate::tracker::{FilterState, ResState, SinState};

// Replaces parts of `waveform` that can be precomputed with their precomputed Fixed versions. Notably,
// infinite waveforms and waveforms that depend on or have dynamic behavior (Slider, Marked, Captured)
// cannot be precomputed. This should be called after remove_seq.
//
// N.B. This isn't currently safe as values at negative positions aren't considered; for example, time is
// negative before zero, but a fixed waveform is always zero before its start.
pub fn precompute(
    generator: &Generator,
    waveform: Waveform<FilterState, SinState, ResState>,
) -> Waveform<FilterState, SinState, ResState> {
    // TODO maybe move this whole thing into Generator?

    enum Result {
        Precomputed(Vec<f32>),
        Infinite(Waveform<FilterState, SinState, ResState>),
        Dynamic(Waveform<FilterState, SinState, ResState>),
    }

    impl Into<Waveform<FilterState, SinState, ResState>> for Result {
        fn into(self) -> Waveform<FilterState, SinState, ResState> {
            match self {
                Result::Precomputed(v) => Waveform::Fixed(v),
                Result::Infinite(w) => w,
                Result::Dynamic(w) => w,
            }
        }
    }

    fn precompute_internal(
        generator: &Generator,
        waveform: Waveform<FilterState, SinState, ResState>,
    ) -> Result {
        use Result::*;
        use Waveform::*;

        // TODO how do we feel about all of the usize::MAX here?
        // TODO there's slightly more repetition than I'd like... for example, lots of "if the child is Dynamic,
        // then parent is Dynamic"
        // Maybe Precomputed should also be a waveform?

        match waveform {
            Const(_) | Time | Noise => Infinite(waveform),
            Fixed(v) => Precomputed(v),
            Fin { length, waveform } => match precompute_internal(generator, *waveform) {
                Precomputed(v) => Precomputed(generator.generate(
                    &Fin {
                        length,
                        waveform: Box::new(Fixed(v)),
                    },
                    0,
                    usize::MAX,
                )),
                Infinite(waveform) => Precomputed(generator.generate(
                    &Fin {
                        length,
                        waveform: Box::new(waveform),
                    },
                    0,
                    usize::MAX,
                )),
                Dynamic(waveform) => {
                    println!(
                        "Cannot precompute Fin because inner waveform is dynamic: {:?}",
                        &waveform
                    );
                    Dynamic(Fin {
                        length,
                        waveform: Box::new(waveform),
                    })
                }
            },
            Seq { .. } => {
                panic!("Seq should have been replaced by replace_seq before precompute is called");
            }
            Append(a, b) => match (
                precompute_internal(generator, *a),
                precompute_internal(generator, *b),
            ) {
                (Precomputed(a), Precomputed(b)) => {
                    let v = [a, b].concat();
                    Precomputed(v)
                }
                (Infinite(a), Infinite(b)) => Infinite(Append(Box::new(a), Box::new(b))),
                (Infinite(a), Precomputed(v)) => Infinite(Append(Box::new(a), Box::new(Fixed(v)))),
                (Precomputed(v), Infinite(b)) => Infinite(Append(Box::new(Fixed(v)), Box::new(b))),
                // At least one is Dynamic
                (a, b) => Dynamic(Append(Box::new(a.into()), Box::new(b.into()))),
            },
            Sin(waveform, state) => match precompute_internal(generator, *waveform) {
                Precomputed(v) => {
                    let v = v.into_iter().map(|x| x.sin()).collect();
                    Precomputed(v)
                }
                Infinite(waveform) => Infinite(Sin(Box::new(waveform), state)),
                Dynamic(waveform) => Dynamic(Sin(Box::new(waveform), state)),
            },
            BinaryPointOp(op, a, b) => match (
                precompute_internal(generator, *a),
                precompute_internal(generator, *b),
            ) {
                (Precomputed(a), Precomputed(b)) => Precomputed(generator.generate(
                    &BinaryPointOp(op, Box::new(Fixed(a)), Box::new(Fixed(b))),
                    0,
                    usize::MAX,
                )),
                (Infinite(a), Infinite(b)) => Infinite(BinaryPointOp(op, Box::new(a), Box::new(b))),
                (Infinite(a), Precomputed(v)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(a), Box::new(Fixed(v))))
                    }
                    Operator::Multiply | Operator::Divide => Precomputed(generator.generate(
                        &BinaryPointOp(op, Box::new(a), Box::new(Fixed(v))),
                        0,
                        usize::MAX,
                    )),
                },
                (Precomputed(v), Infinite(b)) => match op {
                    Operator::Add | Operator::Subtract => {
                        Infinite(BinaryPointOp(op, Box::new(Fixed(v)), Box::new(b)))
                    }
                    Operator::Multiply | Operator::Divide => Precomputed(generator.generate(
                        &BinaryPointOp(op, Box::new(Fixed(v)), Box::new(b)),
                        0,
                        usize::MAX,
                    )),
                },
                // At least one is Dynamic
                (a, b) => Dynamic(BinaryPointOp(op, Box::new(a.into()), Box::new(b.into()))),
            },
            Filter {
                waveform,
                feed_forward,
                feedback,
                state,
            } => {
                match (
                    precompute_internal(generator, *waveform),
                    precompute_internal(generator, *feed_forward),
                    precompute_internal(generator, *feedback),
                ) {
                    (Precomputed(w), Precomputed(ff), Precomputed(fb)) => {
                        let ff_len = ff.len();
                        Precomputed(generator.generate(
                            &Filter {
                                waveform: Box::new(Fixed(w)),
                                feed_forward: Box::new(Fixed(ff)),
                                feedback: Box::new(Fixed(fb)),
                                state,
                            },
                            0,
                            usize::MAX - (ff_len - 1), // because Filter will add to desired
                        ))
                    }
                    (Dynamic(waveform), feed_forward, feedback) => Dynamic(Filter {
                        waveform: Box::new(waveform),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                    (waveform, Dynamic(feed_forward), feedback) => Dynamic(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                    (waveform, feed_forward, Dynamic(feedback)) => Dynamic(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback),
                        state,
                    }),
                    // None are Dynamic, at least one is Infinite
                    (waveform, feed_forward, feedback) => Infinite(Filter {
                        waveform: Box::new(waveform.into()),
                        feed_forward: Box::new(feed_forward.into()),
                        feedback: Box::new(feedback.into()),
                        state,
                    }),
                }
            }
            Res {
                trigger,
                waveform,
                state,
            } => {
                match (
                    precompute_internal(generator, *trigger),
                    precompute_internal(generator, *waveform),
                ) {
                    (Precomputed(t), Precomputed(w)) => Precomputed(generator.generate(
                        &Res {
                            trigger: Box::new(Fixed(t)),
                            waveform: Box::new(Fixed(w)),
                            state,
                        },
                        0,
                        usize::MAX,
                    )),
                    (Dynamic(trigger), waveform) => Dynamic(Res {
                        trigger: Box::new(trigger),
                        waveform: Box::new(waveform.into()),
                        state,
                    }),
                    (trigger, Dynamic(waveform)) => Dynamic(Res {
                        trigger: Box::new(trigger.into()),
                        waveform: Box::new(waveform),
                        state,
                    }),
                    // Neither is Dynamic, at least one is Infinite
                    (trigger, waveform) => Infinite(Res {
                        trigger: Box::new(trigger.into()),
                        waveform: Box::new(waveform.into()),
                        state,
                    }),
                }
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                match (
                    precompute_internal(generator, *trigger),
                    precompute_internal(generator, *positive_waveform),
                    precompute_internal(generator, *negative_waveform),
                ) {
                    (Precomputed(t), Precomputed(p), Precomputed(n)) => {
                        Precomputed(generator.generate(
                            &Alt {
                                trigger: Box::new(Fixed(t)),
                                positive_waveform: Box::new(Fixed(p)),
                                negative_waveform: Box::new(Fixed(n)),
                            },
                            0,
                            usize::MAX,
                        ))
                    }
                    (Dynamic(waveform), positive_waveform, negative_waveform) => Dynamic(Alt {
                        trigger: Box::new(waveform),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                    (waveform, Dynamic(positive_waveform), negative_waveform) => Dynamic(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                    (waveform, positive_waveform, Dynamic(negative_waveform)) => Dynamic(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform),
                    }),
                    // None are Dynamic, at least one is Infinite
                    (waveform, positive_waveform, negative_waveform) => Infinite(Alt {
                        trigger: Box::new(waveform.into()),
                        positive_waveform: Box::new(positive_waveform.into()),
                        negative_waveform: Box::new(negative_waveform.into()),
                    }),
                }
            }
            Slider(_) | Marked { .. } | Captured { .. } => Dynamic(waveform),
        }
    }

    return precompute_internal(generator, waveform).into();
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
                    Box::new(Sin(Box::new(Time), ())),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w2),
            BinaryPointOp(
                Operator::Add,
                Box::new(Sin(Box::new(Time), ())),
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
                    Box::new(Sin(Box::new(Time), ())),
                )),
            )),
            Box::new(Const(5.0)),
        );
        assert_eq!(
            simplify(w3),
            BinaryPointOp(
                Operator::Multiply,
                Box::new(Sin(Box::new(Time), ())),
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
                    Box::new(Sin(Box::new(Time), ())),
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
                    Box::new(Sin(Box::new(Time), ())),
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
                    Box::new(Time),
                    Box::new(Const(-2.0)),
                )),
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(Fin {
                length: Box::new(BinaryPointOp(
                    Operator::Add,
                    Box::new(Time),
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
                    Box::new(Time),
                    Box::new(Const(-1.5))
                )),
                waveform: Box::new(Const(15.0)),
            }
        );
    }
}
