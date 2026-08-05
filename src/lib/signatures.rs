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
        vec![Type::float_or_waveform(), Type::float_or_waveform()],
        Type::waveform(),
    ));
    rows.push(Type::function(
        vec![Type::seq(), Type::float_or_waveform()],
        Type::seq(),
    ));
    rows.push(Type::function(
        vec![Type::float_or_waveform(), Type::seq()],
        Type::seq(),
    ));
    Type::And(rows)
}

/// The result type of curried waveform filters (`fin(len)`, `filter(...)`,
/// `capture(name)`, `mark(id)`): applied to a waveform they produce a
/// waveform, and a seq threads through (`builtins::curry`).
fn waveform_filter() -> Type {
    Type::And(vec![
        Type::function(vec![Type::float_or_waveform()], Type::waveform()),
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
                Type::function(vec![Type::float_or_waveform()], Type::waveform()),
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
            Type::function(
                vec![Type::seq(), Type::float_or_waveform()],
                Type::waveform(),
            ),
        ]),
        "==" | "!=" => Type::Forall(
            vec![0],
            Box::new(Type::function(vec![a(), a()], Type::Bool)),
        ),
        "<" | "<=" | ">" | ">=" => Type::function(vec![Type::float(), Type::float()], Type::Bool),
        "log" => Type::function(vec![Type::float(), Type::float()], Type::float()),
        "sqrt" | "exp" => Type::function(vec![Type::float()], Type::float()),
        // Zero frequency with a constant phase folds to a constant, so the
        // result may be a float or a waveform.
        "sine" => Type::function(
            vec![Type::float_or_waveform(), Type::float_or_waveform()],
            Type::float_or_waveform(),
        ),
        "cos" => Type::function(vec![Type::float_or_waveform()], Type::float_or_waveform()),
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
        // Variadic at run time (one or more lists, or waveforms), but every
        // known call site is binary on lists; other forms warn spuriously.
        "append" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::List(Box::new(a())), Type::List(Box::new(a()))],
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
        "fixed" => Type::function(vec![Type::List(Box::new(Type::float()))], Type::waveform()),
        // Curried built-ins: applying the first argument list returns the
        // filter, which threads seqs through.
        "fin" => Type::function(vec![Type::float_or_waveform()], waveform_filter()),
        "seq" => Type::function(
            vec![Type::float_or_waveform()],
            Type::function(vec![Type::float_or_waveform()], Type::seq()),
        ),
        "unseq" => Type::function(
            vec![],
            Type::function(vec![Type::seq()], Type::float_or_waveform()),
        ),
        "filter" => Type::function(
            vec![
                Type::List(Box::new(Type::float_or_waveform())),
                Type::List(Box::new(Type::float_or_waveform())),
            ],
            waveform_filter(),
        ),
        "reset" => Type::function(
            vec![Type::float_or_waveform(), Type::float_or_waveform()],
            Type::waveform(),
        ),
        "alt" => Type::function(
            vec![
                Type::float_or_waveform(),
                Type::float_or_waveform(),
                Type::float_or_waveform(),
            ],
            Type::waveform(),
        ),
        "capture" => Type::function(vec![Type::String], waveform_filter()),
        "__chord" => Type::function(
            vec![Type::List(Box::new(Type::float_or_waveform()))],
            Type::waveform(),
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
    use super::*;
    use crate::builtins;
    use crate::expr::{Binding, Expr, SourceBinding};
    use crate::types::Sort;

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

    #[test]
    fn append_is_binary_on_lists() {
        let Some(Type::Forall(_, body)) = signature("append") else {
            panic!("append should be polymorphic");
        };
        let Type::Function { positional, .. } = *body else {
            panic!("append should be a function");
        };
        assert_eq!(positional.len(), 2);
    }

    #[test]
    fn unknown_names_have_no_signature() {
        assert_eq!(signature("no_such_builtin"), None);
    }
}
