//! Type signatures for the built-in bindings.
//!
//! Built-ins are opaque Rust closures, so the checker cannot infer their types;
//! this table declares one signature per built-in name. It must be kept in sync
//! with [`crate::builtins::add_bindings`] (a test below checks that every
//! registered built-in has an entry).
//!
//! Signatures are prenex schemes in the sense of "Let Arguments Go First" §3.1
//! (polytypes quantify only at the top), playing the role that typing context
//! entries `x : A` play in the paper's rule AT-Var (Fig. 16). Overloaded
//! built-ins carry intersections of arrows ([`Type::And`]) whose conjuncts
//! mirror the runtime's match arms in order — a declarative table in the style
//! of Freeman and Pfenning's principal types for constructors; selection
//! against the application context happens in [`crate::infer`].
//!
//! Domains are sorts — unions of the numeric atoms — and admit exactly the
//! values the runtime arm accepts; contracts that require true scalars
//! (`unfold`'s count, `nth`'s index, the comparisons) say so with `float`/`int`
//! refinements.

use crate::types::Type;

/// Builds the intersection table shared by the binary arithmetic operators,
/// mirroring `binary_op`'s runtime arms: constants fold (preserving
/// integrality when `int_row`), any waveform involvement builds a waveform,
/// one seq threads its offset, and two seqs match no conjunct (a runtime
/// error).
fn binary_arithmetic(int_row: bool) -> Type {
    let mut rows = Vec::new();
    if int_row {
        rows.push(Type::function(vec![Type::int(), Type::int()], Type::int()));
    }
    rows.push(Type::function(
        vec![Type::float(), Type::float()],
        Type::float(),
    ));
    rows.push(Type::function(
        vec![Type::waveform(), Type::waveform()],
        Type::non_const_wave(),
    ));
    rows.push(Type::function(
        vec![Type::seq(), Type::waveform()],
        Type::seq(),
    ));
    rows.push(Type::function(
        vec![Type::waveform(), Type::seq()],
        Type::seq(),
    ));
    Type::And(rows)
}

/// The result type of curried waveform filters (`fin(len)`, `filter(...)`,
/// `capture(name)`, `mark(id)`): applied to a waveform they produce a
/// waveform, and a seq threads through (`builtins::curry`).
fn waveform_filter() -> Type {
    Type::And(vec![
        Type::function(vec![Type::waveform()], Type::non_const_wave()),
        Type::function(vec![Type::seq()], Type::seq()),
    ])
}

/// Returns the signature of the built-in named `name`, or `None` for names
/// without a declared signature; callers should treat those as
/// [`Type::Dynamic`].
pub fn signature(name: &str) -> Option<Type> {
    // Var ids are local to each signature; instantiation freshens them.
    let a = || Type::Var(0);
    let b = || Type::Var(1);
    let ty = match name {
        "+" | "*" | "&" => binary_arithmetic(true),
        "/" | "pow" => binary_arithmetic(false),
        // Unary and binary rows in one intersection; selection matches the
        // call's arity. Unary minus preserves integrality; there is no
        // unary seq row (`unary_op` has no `Seq` arm).
        "-" => {
            let unary = vec![
                Type::function(vec![Type::int()], Type::int()),
                Type::function(vec![Type::float()], Type::float()),
                Type::function(vec![Type::waveform()], Type::non_const_wave()),
            ];
            let Type::And(binary) = binary_arithmetic(true) else {
                unreachable!("binary_arithmetic returns an intersection");
            };
            Type::And(unary.into_iter().chain(binary).collect())
        }
        // The left operand must be an actual seq; the right may be any
        // numeric. A seq right threads the combined offset (`followed_by`'s
        // `Seq` arm), so the result is again a seq and `\` chains type
        // precisely; anything else ends the chain with a waveform.
        "\\" => Type::And(vec![
            Type::function(vec![Type::seq(), Type::seq()], Type::seq()),
            Type::function(vec![Type::seq(), Type::waveform()], Type::non_const_wave()),
        ]),
        // Comparison is by value on the three kinds `equals` matches, not
        // structural: two waveforms, two seqs, two lists — anything the
        // runtime cannot take apart — have no arm rather than comparing
        // false. Constants compare as the numbers they are, so the numeric
        // row is `float` and not `waveform`.
        "==" | "!=" => Type::And(vec![
            Type::function(vec![Type::float(), Type::float()], Type::Bool),
            Type::function(vec![Type::Bool, Type::Bool], Type::Bool),
            Type::function(vec![Type::String, Type::String], Type::Bool),
        ]),
        "<" | "<=" | ">" | ">=" => Type::function(vec![Type::float(), Type::float()], Type::Bool),
        "log" => Type::function(vec![Type::float(), Type::float()], Type::float()),
        "sqrt" | "exp" => Type::function(vec![Type::float()], Type::float()),
        // Zero frequency with a constant phase folds to a constant, so the
        // result may be a float or a waveform.
        "sine" => Type::function(vec![Type::waveform(), Type::waveform()], Type::waveform()),
        "cos" => Type::function(vec![Type::waveform()], Type::waveform()),
        "map" => Type::Forall(
            vec![0, 1],
            Box::new(Type::function(
                vec![Type::function(vec![a()], b()), Type::List(Box::new(a()))],
                Type::List(Box::new(b())),
            )),
        ),
        "reduce" => Type::Forall(
            vec![0, 1],
            Box::new(Type::function(
                vec![
                    Type::function(vec![b(), a()], b()),
                    b(),
                    Type::List(Box::new(a())),
                ],
                b(),
            )),
        ),
        // The count is hard-checked integral (and non-negative) at runtime.
        "unfold" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::function(vec![a()], a()), a(), Type::int()],
                Type::List(Box::new(a())),
            )),
        ),
        // Exactly the runtime's `Waveform::Append`: two waveforms, end
        // to end. Lists join with `concat`.
        "append" => Type::function(
            vec![Type::waveform(), Type::waveform()],
            Type::non_const_wave(),
        ),
        "concat" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::List(Box::new(Type::List(Box::new(a()))))],
                Type::List(Box::new(a())),
            )),
        ),
        // The index is hard-checked integral (and non-negative) at runtime.
        "nth" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::int(), Type::List(Box::new(a()))],
                a(),
            )),
        ),
        "fixed" => Type::function(
            vec![Type::List(Box::new(Type::float()))],
            Type::non_const_wave(),
        ),
        // Curried built-ins: applying the first argument list returns the
        // filter, which threads seqs through.
        "fin" => Type::function(vec![Type::waveform()], waveform_filter()),
        "seq" => Type::function(
            vec![Type::waveform()],
            Type::function(vec![Type::waveform()], Type::seq()),
        ),
        "unseq" => Type::function(vec![], Type::function(vec![Type::seq()], Type::waveform())),
        "filter" => Type::function(
            vec![
                Type::List(Box::new(Type::waveform())),
                Type::List(Box::new(Type::waveform())),
            ],
            waveform_filter(),
        ),
        "reset" => Type::function(
            vec![Type::waveform(), Type::waveform()],
            Type::non_const_wave(),
        ),
        "alt" => Type::function(
            vec![Type::waveform(), Type::waveform(), Type::waveform()],
            Type::non_const_wave(),
        ),
        "capture" => Type::function(vec![Type::String], waveform_filter()),
        "__chord" => Type::function(
            vec![Type::List(Box::new(Type::waveform()))],
            Type::non_const_wave(),
        ),
        // Every element must be a seq, and a fold of seqs is itself a seq — the
        // empty fold being the empty seq, `\`'s identity.
        "__sequence" => Type::function(vec![Type::List(Box::new(Type::seq()))], Type::seq()),
        // Added by the native prelude rather than `add_bindings`. The mark
        // id is hard-checked integral (and >= 1) at runtime.
        "mark" => Type::function(vec![Type::int()], waveform_filter()),
        // Variadic and heterogeneous; returns its last argument.
        "debug" => Type::Dynamic,
        _ => return None,
    };
    Some(ty)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::builtins;
    use crate::expr::{Binding, BuiltInFn, Expr, Pattern, SourceBinding, boxed};
    use crate::types::{Refinement, Sort};
    use crate::waveform::{Operator, Waveform};

    /// Every built-in registered by `add_bindings` must have a signature, so
    /// new built-ins fail this test until the table above learns about them.
    #[test]
    fn signatures_cover_registered_builtins() {
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        for binding in &bindings {
            if let Binding::Definition(_, expr) = &binding.binding {
                if let Expr::BuiltIn { name, .. } = &expr.expr {
                    assert!(
                        signature(name).is_some(),
                        "built-in \"{}\" has no signature",
                        name
                    );
                }
            }
        }
    }

    #[test]
    fn prelude_only_builtins_have_signatures() {
        // These are bound by the native prelude, not `add_bindings`.
        assert!(signature("mark").is_some());
        assert!(signature("debug").is_some());
    }

    /// The intersection rows must mirror `binary_op`'s runtime arms: no
    /// conjunct accepts two seqs, exactly one seq threads through, and
    /// constants fold (preserving integrality for `+`).
    #[test]
    fn arithmetic_tables_mirror_the_runtime() {
        let Some(Type::And(rows)) = signature("+") else {
            panic!("+ should be an intersection");
        };
        assert_eq!(
            rows[0],
            Type::function(vec![Type::int(), Type::int()], Type::int())
        );
        for row in &rows {
            let Type::Function { positional, .. } = row else {
                panic!("conjuncts are arrows");
            };
            let both_admit_seq = positional.iter().all(|domain| {
                matches!(domain, Type::Numeric(crate::types::Refinement::Ground(sort))
                    if !sort.intersect(Sort::SEQ).is_empty())
            });
            assert!(!both_admit_seq, "no row may accept two seqs");
        }
        // `-` carries unary and binary rows in one intersection.
        let Some(Type::And(rows)) = signature("-") else {
            panic!("- should be an intersection");
        };
        assert!(
            rows.iter().any(
                |row| matches!(row, Type::Function { positional, .. } if positional.len() == 1)
            )
        );
        assert!(
            rows.iter().any(
                |row| matches!(row, Type::Function { positional, .. } if positional.len() == 2)
            )
        );
        // `/` has no int row: integers divide to floats.
        let Some(Type::And(rows)) = signature("/") else {
            panic!("/ should be an intersection");
        };
        assert_eq!(
            rows[0],
            Type::function(vec![Type::float(), Type::float()], Type::float())
        );
    }

    /// Equality's table is exactly the arms `equals` matches, with no
    /// polymorphic row — which is also what lets the conformance harness
    /// below exercise it, since that skips rows with non-ground domains.
    #[test]
    fn equality_tables_mirror_the_runtime() {
        for name in ["==", "!="] {
            let Some(Type::And(rows)) = signature(name) else {
                panic!("{} should be an intersection", name);
            };
            assert_eq!(
                rows,
                vec![
                    Type::function(vec![Type::float(), Type::float()], Type::Bool),
                    Type::function(vec![Type::Bool, Type::Bool], Type::Bool),
                    Type::function(vec![Type::String, Type::String], Type::Bool),
                ],
                "for {}",
                name
            );
        }
    }

    #[test]
    fn append_takes_two_waveforms() {
        assert_eq!(
            signature("append"),
            Some(Type::function(
                vec![Type::waveform(), Type::waveform()],
                Type::non_const_wave(),
            ))
        );
    }

    #[test]
    fn unknown_names_have_no_signature() {
        assert_eq!(signature("no_such_builtin"), None);
    }

    /// One runtime value inhabiting each atom, for driving the built-ins.
    fn representative(atom: Sort) -> Expr<u32, ()> {
        if atom == Sort::INT {
            Expr::float(2.0)
        } else if atom == Sort::NON_INT_ONLY {
            Expr::float(0.5)
        } else if atom == Sort::NON_CONST_WAVE {
            Expr::Waveform(Waveform::Time(()))
        } else if atom == Sort::SEQ {
            // A realistic seq: offset linear in time (`time - 1`), so
            // offset-threading arms (`\`'s, `add_offsets`) work.
            Expr::Seq {
                offset: boxed(Expr::Waveform(Waveform::BinaryPointOp(
                    Operator::Subtract,
                    Box::new(Waveform::Time(())),
                    Box::new(Waveform::Const(1.0)),
                ))),
                waveform: boxed(Expr::Waveform(Waveform::Time(()))),
            }
        } else {
            panic!("not an atom: {}", atom)
        }
    }

    /// The runtime sort of a result; `None` for non-numeric values.
    fn sort_of(expr: &Expr<u32, ()>) -> Option<Sort> {
        match expr {
            Expr::Waveform(Waveform::Const(value)) => Some(if value.fract() == 0.0 {
                Sort::INT
            } else {
                Sort::NON_INT_ONLY
            }),
            Expr::Waveform(_) => Some(Sort::NON_CONST_WAVE),
            Expr::Seq { .. } => Some(Sort::SEQ),
            _ => None,
        }
    }

    /// The rows of `ty` at `arity` whose domains are all numeric grounds.
    fn numeric_rows(ty: &Type, arity: usize) -> Vec<(Vec<Sort>, Type)> {
        let conjuncts: Vec<&Type> = match ty {
            Type::And(rows) => rows.iter().collect(),
            other => vec![other],
        };
        let mut rows = Vec::new();
        for conjunct in conjuncts {
            let Type::Function {
                positional,
                named,
                result,
            } = conjunct
            else {
                continue;
            };
            if positional.len() != arity || !named.is_empty() {
                continue;
            }
            let domains: Option<Vec<Sort>> = positional
                .iter()
                .map(|domain| match domain {
                    Type::Numeric(Refinement::Ground(sort)) => Some(*sort),
                    _ => None,
                })
                .collect();
            if let Some(domains) = domains {
                rows.push((domains, (**result).clone()));
            }
        }
        rows
    }

    /// The distinct arities of `ty`'s arrow conjuncts.
    fn arities(ty: &Type) -> Vec<usize> {
        let conjuncts: Vec<&Type> = match ty {
            Type::And(rows) => rows.iter().collect(),
            other => vec![other],
        };
        let mut arities = Vec::new();
        for conjunct in conjuncts {
            if let Type::Function { positional, .. } = conjunct
                && !arities.contains(&positional.len())
            {
                arities.push(positional.len());
            }
        }
        arities
    }

    /// Conformance of one callable to its table at one arity, mirroring
    /// selection semantics: every atom vector covered by some row must
    /// evaluate without error to a value within the *first* covering
    /// row's result (table order is most-specific-first, as selection
    /// reads it), and every uncovered vector must error. Curried results
    /// recurse on the returned callable.
    fn conform(name: &str, function: &BuiltInFn<u32, ()>, ty: &Type, arity: usize) {
        let rows = numeric_rows(ty, arity);
        if rows.is_empty() {
            return;
        }
        let atoms = [
            Sort::INT,
            Sort::NON_INT_ONLY,
            Sort::NON_CONST_WAVE,
            Sort::SEQ,
        ];
        for index in 0..atoms.len().pow(arity as u32) {
            let vector: Vec<Sort> = (0..arity)
                .map(|position| atoms[(index / atoms.len().pow(position as u32)) % atoms.len()])
                .collect();
            let shown: Vec<String> = vector.iter().map(Sort::to_string).collect();
            let governing = rows.iter().find(|(domains, _)| {
                vector
                    .iter()
                    .zip(domains)
                    .all(|(atom, domain)| atom.is_subset(*domain))
            });
            let arguments: Vec<Expr<u32, ()>> =
                vector.iter().map(|atom| representative(*atom)).collect();
            let result = function.0(arguments);
            let Some((_, declared)) = governing else {
                assert!(
                    matches!(result, Expr::Error(_)),
                    "{} accepted uncovered ({}): {}",
                    name,
                    shown.join(", "),
                    result
                );
                continue;
            };
            assert!(
                !matches!(result, Expr::Error(_)),
                "{} errored on covered ({}): {}",
                name,
                shown.join(", "),
                result
            );
            match declared {
                Type::Numeric(Refinement::Ground(sort)) => {
                    let actual = sort_of(&result).unwrap_or_else(|| {
                        panic!(
                            "{} on ({}) returned a non-numeric {}",
                            name,
                            shown.join(", "),
                            result
                        )
                    });
                    assert!(
                        actual.is_subset(*sort),
                        "{} on ({}): result sort {} outside declared {}",
                        name,
                        shown.join(", "),
                        actual,
                        sort
                    );
                }
                declared @ (Type::Function { .. } | Type::And(_)) => {
                    let Expr::BuiltIn { function, .. } = &result else {
                        panic!(
                            "{} on ({}) should curry, returned {}",
                            name,
                            shown.join(", "),
                            result
                        )
                    };
                    for arity in arities(declared) {
                        conform(&format!("{}(...)", name), function, declared, arity);
                    }
                }
                // Bool and friends: evaluating without error suffices.
                _ => {}
            }
        }
    }

    /// Every built-in with numeric-ground rows conforms to them — the
    /// signature-faithfulness obligation of the sound configuration. Rows
    /// over structural domains (lists, functions, ∀-polymorphic) are out
    /// of scope here; `mark` is prelude-native and `debug` is Dynamic.
    #[test]
    fn signatures_conform_to_the_runtime() {
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        let mut functions: HashMap<String, BuiltInFn<u32, ()>> = HashMap::new();
        for binding in bindings {
            if let Binding::Definition(Pattern::Identifier(name), expr) = binding.binding
                && let Expr::BuiltIn { function, .. } = expr.expr
            {
                functions.insert(name, function);
            }
        }
        let mut conformed = 0;
        for (name, function) in &functions {
            let Some(ty) = signature(name) else {
                continue;
            };
            for arity in arities(&ty) {
                if !numeric_rows(&ty, arity).is_empty() {
                    conformed += 1;
                    conform(name, function, &ty, arity);
                }
            }
        }
        // The harness must actually be exercising the operator tables.
        assert!(conformed >= 15, "only {} tables conformed", conformed);
    }
}
