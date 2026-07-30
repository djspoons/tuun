//! Type signatures for the built-in bindings.
//!
//! Built-ins are opaque Rust closures, so the checker cannot infer their
//! types; this table declares one signature per built-in name. It must be
//! kept in sync with [`crate::builtins::add_bindings`] (a test below checks
//! that every registered built-in has an entry).
//!
//! Signatures are prenex schemes in the sense of "Let Arguments Go First"
//! §3.1 (polytypes quantify only at the top), playing the role that typing
//! context entries `x : A` play in the paper's rule AT-Var (Fig. 16).
//!
//! Signatures deliberately simplify: each built-in gets a single type, with the
//! subtypings `float <: waveform` and `seq <: waveform`. Where the run-time
//! behavior is wider than the signature (e.g. `append` on waveforms), the
//! checker reports a spurious warning; where it is narrower (e.g. `1 + 2`
//! producing a float), the checker reports the wider type. Both are accepted
//! costs of the single-signature design.

use crate::types::Type;

/// Returns the signature of the built-in named `name`.
///
/// `arity` is the number of positional arguments at the call site, when
/// known; it selects between forms for arity-overloaded built-ins (unary
/// versus binary `-`). Returns `None` for names without a declared
/// signature; callers should treat those as [`Type::Dynamic`].
pub fn signature(name: &str, arity: Option<usize>) -> Option<Type> {
    // Var ids are local to each signature; instantiation freshens them.
    let a = || Type::Var(0);
    let b = || Type::Var(1);
    let binary_wave = || Type::function(vec![Type::Waveform, Type::Waveform], Type::Waveform);
    let comparison = || Type::function(vec![Type::Float, Type::Float], Type::Bool);
    let ty = match name {
        "+" | "*" | "/" | "&" | "pow" => binary_wave(),
        "-" => match arity {
            Some(1) => Type::function(vec![Type::Waveform], Type::Waveform),
            _ => binary_wave(),
        },
        "\\" => Type::function(vec![Type::Seq, Type::Waveform], Type::Seq),
        "==" | "!=" => Type::Forall(
            vec![0],
            Box::new(Type::function(vec![a(), a()], Type::Bool)),
        ),
        "<" | "<=" | ">" | ">=" => comparison(),
        "log" => Type::function(vec![Type::Float, Type::Float], Type::Float),
        "sqrt" | "exp" => Type::function(vec![Type::Float], Type::Float),
        "sine" => binary_wave(),
        "cos" => Type::function(vec![Type::Waveform], Type::Waveform),
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
        "unfold" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::function(vec![a()], a()), a(), Type::Float],
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
        "nth" => Type::Forall(
            vec![0],
            Box::new(Type::function(
                vec![Type::Float, Type::List(Box::new(a()))],
                a(),
            )),
        ),
        "fixed" => Type::function(vec![Type::List(Box::new(Type::Float))], Type::Waveform),
        // Curried built-ins: applying the first argument list returns another
        // built-in expecting the waveform.
        "fin" => Type::function(
            vec![Type::Waveform],
            Type::function(vec![Type::Waveform], Type::Waveform),
        ),
        "seq" => Type::function(
            vec![Type::Waveform],
            Type::function(vec![Type::Waveform], Type::Seq),
        ),
        "unseq" => Type::function(vec![], Type::function(vec![Type::Seq], Type::Waveform)),
        "filter" => Type::function(
            vec![
                Type::List(Box::new(Type::Waveform)),
                Type::List(Box::new(Type::Waveform)),
            ],
            Type::function(vec![Type::Waveform], Type::Waveform),
        ),
        "reset" => binary_wave(),
        "alt" => Type::function(
            vec![Type::Waveform, Type::Waveform, Type::Waveform],
            Type::Waveform,
        ),
        "capture" => Type::function(
            vec![Type::String],
            Type::function(vec![Type::Waveform], Type::Waveform),
        ),
        "__chord" => Type::function(vec![Type::List(Box::new(Type::Waveform))], Type::Waveform),
        // The run-time contract (all elements but the last must be seqs) is
        // not expressible as a single signature; `[waveform]` accepts the
        // mixed lists that real programs build.
        "__sequence" => Type::function(vec![Type::List(Box::new(Type::Waveform))], Type::Seq),
        // Added by the native prelude rather than `add_bindings`.
        "mark" => Type::function(
            vec![Type::Float],
            Type::function(vec![Type::Waveform], Type::Waveform),
        ),
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
                        signature(name, None).is_some(),
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
        assert!(signature("mark", None).is_some());
        assert!(signature("debug", None).is_some());
    }

    #[test]
    fn arity_selects_unary_minus() {
        assert_eq!(
            signature("-", Some(1)),
            Some(Type::function(vec![Type::Waveform], Type::Waveform))
        );
        assert_eq!(
            signature("-", Some(2)),
            Some(Type::function(
                vec![Type::Waveform, Type::Waveform],
                Type::Waveform
            ))
        );
    }

    #[test]
    fn append_is_binary_on_lists() {
        let Some(Type::Forall(_, body)) = signature("append", None) else {
            panic!("append should be polymorphic");
        };
        let Type::Function { positional, .. } = *body else {
            panic!("append should be a function");
        };
        assert_eq!(positional.len(), 2);
    }

    #[test]
    fn unknown_names_have_no_signature() {
        assert_eq!(signature("no_such_builtin", None), None);
    }
}
