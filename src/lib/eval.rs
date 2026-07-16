//! Evaluation of tuun expressions: builds a context from bindings (resolving
//! `open`s through a caller-supplied module resolver), substitutes it into an
//! expression, and reduces the result to a value.

use std::fmt;
use std::fmt::{Debug, Display};

use crate::expr::{Binding, Error, Expr, NamedExprs, Pattern, SourceBinding, SourceExpr};

/// A context entry: a bound name and the closed value it evaluates to.
type ContextEntry<M, S> = (String, SourceExpr<M, S>);

/// Extends the context with a binding for each identifier in the pattern that is bound to
/// itself.
fn extend_with_trivial_context<M, S>(
    context: &mut Vec<(String, SourceExpr<M, S>)>,
    pattern: &Pattern,
) {
    match pattern {
        Pattern::Identifier(name) => {
            context.push((name.clone(), SourceExpr::variable(name.clone())));
        }
        Pattern::Tuple(patterns) => {
            for pattern in patterns {
                extend_with_trivial_context(context, pattern);
            }
        }
    }
}

/// Substitutes any occurrences of the variables in `context` that are found in
/// `expr` with the corresponding expressions. All of the expressions in
/// `context` should be closed values. The resulting expression will be closed.
///
/// Rebuilt nodes keep their spans as provenance (a substituted variable keeps
/// the span of the value's defining expression), so the result's spans locate
/// where each part originated but no longer promise verbatim source text —
/// see the span contract on [`SourceExpr`].
fn substitute<M, S>(
    context: &[(String, SourceExpr<M, S>)],
    expr: SourceExpr<M, S>,
) -> SourceExpr<M, S>
where
    M: Clone,
    S: Clone,
{
    use Expr::{
        Application, Bool, BuiltIn, Error, Float, Function, IfThenElse, List, String, Tuple,
        Variable,
    };
    let SourceExpr { expr, span } = expr;
    match expr {
        Bool(_) | Float(_) | String(_) => SourceExpr { expr, span },
        Expr::Waveform(w) => SourceExpr {
            expr: Expr::Waveform(w),
            span,
        },
        Expr::Seq { offset, waveform } => SourceExpr {
            expr: Expr::Seq {
                offset: Box::new(substitute(context, *offset)),
                waveform: Box::new(substitute(context, *waveform)),
            },
            span,
        },
        Function {
            positional,
            named,
            body,
        } => {
            // Named defaults are evaluated in the enclosing scope: they see
            // the incoming context, not the parameters.
            let named: NamedExprs<M, S> = named
                .into_iter()
                .map(|(name, value)| (name, substitute(context, value)))
                .collect();
            let mut context = Vec::from(context);
            for param in &positional {
                extend_with_trivial_context(&mut context, param);
            }
            for (name, _) in &named {
                context.push((name.clone(), SourceExpr::variable(name.clone())));
            }
            let body = substitute(&context, *body);
            SourceExpr {
                expr: Expr::Function {
                    positional,
                    named,
                    body: Box::new(body),
                },
                span,
            }
        }
        BuiltIn { name, function } => SourceExpr {
            expr: Expr::BuiltIn { name, function },
            span,
        },
        Variable(name) => {
            for (var_name, value) in context.iter().rev() {
                if var_name == &name {
                    return value.clone();
                }
            }
            SourceExpr {
                expr: Error(format!("Variable '{}' not found in context", name)),
                span,
            }
        }
        IfThenElse {
            condition,
            then,
            else_,
        } => {
            let condition = Box::new(substitute(context, *condition));
            let then = Box::new(substitute(context, *then));
            let else_ = Box::new(substitute(context, *else_));
            SourceExpr {
                expr: Expr::IfThenElse {
                    condition,
                    then,
                    else_,
                },
                span,
            }
        }
        Application {
            function,
            positional,
            named,
        } => {
            let function = substitute(context, *function);
            let positional = positional
                .into_iter()
                .map(|a| substitute(context, a))
                .collect();
            let named = named
                .into_iter()
                .map(|(name, value)| (name, substitute(context, value)))
                .collect();
            SourceExpr {
                expr: Expr::Application {
                    function: Box::new(function),
                    positional,
                    named,
                },
                span,
            }
        }
        Tuple(exprs) => SourceExpr {
            expr: Expr::Tuple(exprs.into_iter().map(|e| substitute(context, e)).collect()),
            span,
        },
        List(exprs) => SourceExpr {
            expr: Expr::List(exprs.into_iter().map(|e| substitute(context, e)).collect()),
            span,
        },
        Error(s) => SourceExpr {
            expr: Error(s),
            span,
        },
    }
}

fn extend_context<M, S>(
    context: &mut Vec<(String, SourceExpr<M, S>)>,
    pattern: &Pattern,
    argument: &SourceExpr<M, S>,
) -> Result<(), Error<S>>
where
    M: Clone + Debug + Display,
    S: Clone + Debug,
{
    match (pattern, &argument.expr) {
        (Pattern::Identifier(name), _) => {
            context.push((name.clone(), argument.clone()));
            Ok(())
        }
        (Pattern::Tuple(patterns), Expr::Tuple(arguments)) => {
            if patterns.len() != arguments.len() {
                return Err(Error::with_span(
                    format!(
                        "Mismatched number of elements in pattern {} and arguments {}",
                        pattern, argument
                    ),
                    argument.span.clone(),
                ));
            }
            for (pattern, argument) in patterns.iter().zip(arguments) {
                extend_context(context, pattern, argument)?;
            }
            Ok(())
        }
        _ => Err(Error::with_span(
            format!(
                "Pattern {} does not match actual expression {}",
                pattern, argument.expr
            ),
            argument.span.clone(),
        )),
    }
}

/// Evaluates a closed expression to a value. Closed expressions do not contain
/// variables.
///
/// Equivalent to [`evaluate`] with no bindings and a resolver that always
/// fails, but skips the needless substitution pass.
///
/// The result's spans are provenance (see [`SourceExpr`]): an expression that
/// is already a value keeps its span, an extracted sub-value (a chosen
/// branch, an applied function's body) keeps its own, and a built-in
/// application's result carries the span of the call site.
pub fn evaluate_closed<M, S>(expr: SourceExpr<M, S>) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + fmt::Display + fmt::Debug,
    S: Clone + fmt::Debug,
{
    use Expr::{
        Application, Bool, BuiltIn, Float, Function, IfThenElse, List, Seq, String, Tuple,
        Variable, Waveform,
    };
    let SourceExpr { expr, span } = expr;
    match expr {
        Bool(_) | Float(_) | String(_) | Waveform(_) => {
            // For values, we can preserve the input span, since it still faithfully
            // represents the value.
            Ok(SourceExpr { expr, span })
        }
        Function {
            positional,
            named,
            body,
        } => {
            // Named defaults are evaluated once, here — when the function
            // expression itself is reduced to a value (for a binding, at
            // definition time) — not at each application. Idempotent:
            // defaults that are already values evaluate to themselves.
            let named = named
                .into_iter()
                .map(|(name, value)| Ok((name, evaluate_closed(value)?)))
                .collect::<Result<NamedExprs<M, S>, Error<S>>>()?;
            Ok(SourceExpr {
                expr: Function {
                    positional,
                    named,
                    body,
                },
                span,
            })
        }
        Variable(name) => Err(Error::with_span(
            format!("Variable '{}' not found in context", name),
            span,
        )),
        Seq { offset, waveform } => {
            let offset = evaluate_closed(*offset)?;
            let waveform = evaluate_closed(*waveform)?;
            Ok(SourceExpr {
                expr: Seq {
                    offset: Box::new(offset),
                    waveform: Box::new(waveform),
                },
                span,
            })
        }
        BuiltIn { name, function } => Ok(SourceExpr {
            expr: BuiltIn { name, function },
            span,
        }),
        IfThenElse {
            condition,
            then,
            else_,
        } => {
            // Evaluation may drop the condition's span, so capture it up
            // front to point the error at the original condition.
            let condition_span = condition.span.clone();
            match evaluate_closed(*condition)?.expr {
                Bool(true) => evaluate_closed(*then),
                Bool(false) => evaluate_closed(*else_),
                _ => Err(Error::with_span(
                    "Expected boolean condition".to_string(),
                    condition_span,
                )),
            }
        }
        Application {
            function,
            positional,
            named,
        } => {
            let function = evaluate_closed(*function)?;
            let pos_args = positional
                .into_iter()
                .map(evaluate_closed)
                .collect::<Result<Vec<_>, _>>()?;
            let named = named
                .into_iter()
                .map(|(name, value)| Ok((name, evaluate_closed(value)?)))
                .collect::<Result<NamedExprs<M, S>, Error<S>>>()?;
            match (function.expr, pos_args) {
                (
                    Function {
                        positional: pos_params,
                        named: defaults,
                        body,
                    },
                    pos_args,
                ) => {
                    // Every call-site name must be a declared named
                    // parameter and may appear at most once. (The parser
                    // reports both violations too; these checks defend
                    // synthesized trees.)
                    for (i, (name, _)) in named.iter().enumerate() {
                        if named[..i].iter().any(|(n, _)| n == name) {
                            return Err(Error::with_span(
                                format!("named parameter \"{}\" appears more than once", name),
                                span.clone(),
                            ));
                        }
                        if !defaults.iter().any(|(n, _)| n == name) {
                            return Err(Error::with_span(
                                format!("no named parameter \"{}\"", name),
                                span.clone(),
                            ));
                        }
                    }
                    if pos_args.len() > pos_params.len() {
                        return Err(Error::with_span(
                            "extra positional parameter".to_string(),
                            span.clone(),
                        ));
                    }
                    if pos_args.len() < pos_params.len() {
                        return Err(Error::with_span(
                            format!("missing parameter \"{}\"", pos_params[pos_args.len()]),
                            span.clone(),
                        ));
                    }
                    let mut context = Vec::new();
                    for (param, argument) in pos_params.iter().zip(&pos_args) {
                        extend_context(&mut context, param, argument)?;
                    }
                    // Each named parameter binds the call-site override if
                    // present, else its (already evaluated) default.
                    for (name, default) in &defaults {
                        let value = named
                            .iter()
                            .find(|(n, _)| n == name)
                            .map(|(_, v)| v.clone())
                            .unwrap_or_else(|| default.clone());
                        context.push((name.clone(), value));
                    }
                    let body = substitute(&context, *body);
                    evaluate_closed(body)
                }
                (BuiltIn { name, function }, arguments) => {
                    if let Some((named_name, _)) = named.first() {
                        return Err(Error::with_span(
                            format!(
                                "named argument \"{}\" is not supported by built-in \"{}\"",
                                named_name, name
                            ),
                            span.clone(),
                        ));
                    }
                    // Builtins operate on bare `Expr<M>` values, so unwrap
                    // the SourceExpr children before calling and wrap the
                    // result. Use the outer Application's span (`span`) so
                    // errors — and the result's provenance — point at the
                    // whole `f(x, y)` call site; builtins themselves don't
                    // see spans.
                    let actuals: Vec<Expr<M, S>> = arguments.into_iter().map(|s| s.expr).collect();
                    let result = function.0(actuals);
                    match result {
                        Expr::Error(s) => Err(Error::with_span(s, span.clone())),
                        _ => Ok(SourceExpr { expr: result, span }),
                    }
                }
                (function, _) => Err(Error::with_span(
                    format!("Invalid application: {}", function),
                    span.clone(),
                )),
            }
        }
        Tuple(exprs) => Ok(SourceExpr {
            expr: Tuple(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            span,
        }),
        List(exprs) => Ok(SourceExpr {
            expr: List(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            span,
        }),
        Expr::Error(s) => Err(Error::with_span(s, span)),
    }
}

/// Evaluates `bindings` and then `expr` in the context of those bindings.
///
/// `resolve` should return the parsed but unevaluated bindings of a module.
/// For each `Open(path)`, `resolve` is called with the module path to
/// retrieve that module's parsed bindings, which are then recursively
/// processed.
///
/// After `bindings` is fully processed, `expr` is substituted against the
/// final context and reduced to a value.
pub fn evaluate<'a, M, S, F>(
    resolve: F,
    bindings: &'a [SourceBinding<M, S>],
    expr: SourceExpr<M, S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug,
    S: Clone + Debug,
{
    let mut context: Vec<(String, SourceExpr<M, S>)> = Vec::new();
    build_context(&resolve, bindings, &mut context)?;
    evaluate_with_context(&context, expr)
}

/// Substitutes `context` into `expr` and reduces the result to a value.
fn evaluate_with_context<M, S>(
    context: &[(String, SourceExpr<M, S>)],
    expr: SourceExpr<M, S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + Display + Debug,
    S: Clone + Debug,
{
    let expr = substitute(context, expr);
    evaluate_closed(expr)
}

/// Walks `bindings`, accumulating evaluated entries into `context`.
fn build_context<'a, M, S, F>(
    resolve: &F,
    bindings: &'a [SourceBinding<M, S>],
    context: &mut Vec<ContextEntry<M, S>>,
) -> Result<Vec<ContextEntry<M, S>>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug,
    S: Clone + Debug,
{
    let mut own = Vec::new();
    for source_binding in bindings {
        match &source_binding.binding {
            Binding::Open(path) => {
                let module = resolve(path)?;
                let mut module_context = Vec::new();
                let exports = build_context(resolve, module, &mut module_context)?;
                context.extend(exports);
            }
            Binding::Definition(pattern, def_expr) => {
                let substituted = substitute(context, def_expr.clone());
                let value = evaluate_closed(substituted)?;
                let before = context.len();
                extend_context(context, pattern, &value)?;
                own.extend_from_slice(&context[before..]);
            }
            // Empty bindings have no semantic content — they exist only to
            // anchor annotation/comment spans for source preservation.
            Binding::Empty => {}
        }
    }
    Ok(own)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins;
    use crate::parser::{parse_module, parse_program};

    /// Parses and evaluates `input` with the built-ins in scope.
    fn eval_with_builtins(input: &str) -> Result<SourceExpr<u32, ()>, Error<()>> {
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        evaluate(
            |_: &[String]| Err(Error::new("no modules".to_string())),
            &bindings,
            expr,
        )
    }

    #[test]
    fn test_named_arguments() {
        let f = "let f = fn(x, y = 10) => x * y + 1 in ";
        let run = |call: &str| eval_with_builtins(&format!("{}{}", f, call));
        assert_eq!(format!("{}", run("f(2)").unwrap()), "21");
        assert_eq!(format!("{}", run("f(2, y = 5)").unwrap()), "11");
        assert_eq!(
            run("f(2, 3)").unwrap_err().message(),
            "extra positional parameter"
        );
        assert_eq!(
            run("f(2, z = 3)").unwrap_err().message(),
            "no named parameter \"z\""
        );
        assert_eq!(
            run("f(y = 2)").unwrap_err().message(),
            "missing parameter \"x\""
        );

        // All-named functions can be called with no arguments at all.
        let g = "let g = fn(y = 1) => y in ";
        assert_eq!(
            format!("{}", eval_with_builtins(&format!("{}g()", g)).unwrap()),
            "1"
        );
        assert_eq!(
            format!("{}", eval_with_builtins(&format!("{}g(y = 3)", g)).unwrap()),
            "3"
        );

        // Defaults close over the enclosing scope...
        assert_eq!(
            format!(
                "{}",
                eval_with_builtins("let a = 5, f = fn(x, y = a * 2) => x + y in f(1)").unwrap()
            ),
            "11"
        );
        // ...while the parameter name shadows outer bindings in the body.
        assert_eq!(
            format!(
                "{}",
                eval_with_builtins("let y = 100, f = fn(x, y = 10) => x * y in f(2)").unwrap()
            ),
            "20"
        );

        // Destructuring positional params combine with named ones.
        let h = "let f = fn((a, b), y = 1) => a + b + y in ";
        assert_eq!(
            format!(
                "{}",
                eval_with_builtins(&format!("{}f((1, 2))", h)).unwrap()
            ),
            "4"
        );
        assert_eq!(
            format!(
                "{}",
                eval_with_builtins(&format!("{}f((1, 2), y = 10)", h)).unwrap()
            ),
            "13"
        );

        // Builtins do not take named arguments.
        let error = eval_with_builtins("sine(440, y = 1)").unwrap_err();
        assert!(error.message().contains("built-in \"sine\""), "{}", error);
    }

    #[test]
    fn test_named_defaults_evaluate_once() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let printed: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&printed);
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        bindings.push(
            Binding::Definition(
                Pattern::Identifier("debug".to_string()),
                builtins::debug(move |line| sink.borrow_mut().push(line.to_string())),
            )
            .into(),
        );
        let resolve = |_: &[String]| Err(Error::new("no modules".to_string()));

        // The default is evaluated once, when the function value is
        // created — not at each of the three calls.
        let expr = parse_program::<u32, _>(
            "let f = fn(x, y = debug(1)) => x, _ = f(1), _ = f(2) in f(3)",
            (),
        )
        .unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "3");
        assert_eq!(printed.borrow().as_slice(), ["debug: [1]"]);

        // Even a function that is never applied evaluates its defaults.
        printed.borrow_mut().clear();
        let expr = parse_program::<u32, _>("let f = fn(x, y = debug(1)) => x in 0", ()).unwrap();
        evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(printed.borrow().as_slice(), ["debug: [1]"]);
    }

    #[test]
    fn test_opens_are_scoped() {
        let (b, errors) = parse_module::<u32, _>("two = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let (a, errors) = parse_module::<u32, _>("open b; alias = two;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["a"] {
                Ok(&a[..])
            } else if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::new(format!("no module {:?}", path)))
            }
        };
        let (bindings, errors) = parse_module::<u32, _>("open a;", ()).unwrap();
        assert!(errors.is_empty());

        // `a`'s own definitions can use the names `a` opened...
        let expr = parse_program::<u32, _>("alias", ()).unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "2");

        // ...but opening `a` does not re-export what `a` merely opened.
        let expr = parse_program::<u32, _>("two", ()).unwrap();
        let error = evaluate(resolve, &bindings, expr).unwrap_err();
        assert_eq!(error.message(), "Variable 'two' not found in context");
    }

    #[test]
    fn test_application_arity_is_exact() {
        let expr = parse_program::<u32, _>("(fn(x) => x)(2, 3)", ()).unwrap();
        let error = evaluate_closed(expr).unwrap_err();
        assert_eq!(error.message(), "extra positional parameter");

        let expr = parse_program::<u32, _>("(fn(x, y) => x)(2)", ()).unwrap();
        let error = evaluate_closed(expr).unwrap_err();
        assert_eq!(error.message(), "missing parameter \"y\"");

        // A tuple literal is one argument; destructuring params accept it...
        let expr = parse_program::<u32, _>("(fn((y, z)) => (z, y))((4, 5))", ()).unwrap();
        let evaluated = evaluate_closed(expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(5, 4)");

        // ...and passing two arguments instead is an arity error, not a splat.
        let expr = parse_program::<u32, _>("(fn((y, z)) => y)(4, 5)", ()).unwrap();
        let error = evaluate_closed(expr).unwrap_err();
        assert_eq!(error.message(), "extra positional parameter");
    }

    #[test]
    fn test_function_eval() {
        let input = "(fn(x) => fn(x) => x)(7)(5)";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        println!("Parsed expression: {}", expr);
        let evaluated = evaluate_closed(expr).unwrap();
        assert_eq!(format!("{}", evaluated), "5");

        let input = "(fn(x) => fn(y, z) => (x, y, z))(3)(4, 5)";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        let evaluated = evaluate_closed(expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");

        let input = "(fn(x, (y, z)) => (x, y, z))(3, (4, 5))";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        let evaluated = evaluate_closed(expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");
    }
}
