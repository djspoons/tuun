//! Evaluation of tuun expressions: builds a context from bindings (resolving
//! `open`s through a caller-supplied module resolver), substitutes it into an
//! expression, and reduces the result to a value.

use std::fmt;
use std::fmt::{Debug, Display};

use crate::expr::{Binding, Error, Expr, Pattern, SourceBinding, SourceExpr};

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
        Expr::Seq { offset, waveform } => SourceExpr::from(Expr::Seq {
            offset: Box::new(substitute(context, *offset)),
            waveform: Box::new(substitute(context, *waveform)),
        }),
        Function { pattern, body } => {
            let mut context = Vec::from(context);
            extend_with_trivial_context(&mut context, &pattern);
            let body = substitute(&context, *body);
            SourceExpr::from(Expr::Function {
                pattern,
                body: Box::new(body),
            })
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
            SourceExpr::from(Expr::IfThenElse {
                condition,
                then,
                else_,
            })
        }
        Application { function, argument } => {
            let function = substitute(context, *function);
            let argument = substitute(context, *argument);
            SourceExpr::from(Expr::Application {
                function: Box::new(function),
                argument: Box::new(argument),
            })
        }
        Tuple(exprs) => SourceExpr::from(Expr::Tuple(
            exprs.into_iter().map(|e| substitute(context, e)).collect(),
        )),
        List(exprs) => SourceExpr::from(Expr::List(
            exprs.into_iter().map(|e| substitute(context, e)).collect(),
        )),
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
/// The resulting expression will have a `span` only if the argument is a value
/// already.
fn evaluate_closed<M, S>(expr: SourceExpr<M, S>) -> Result<SourceExpr<M, S>, Error<S>>
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
        Bool(_) | Float(_) | String(_) | Waveform(_) | Function { .. } => {
            // For values, we can preserve the input span, since it still faithfully
            // represents the value.
            // TODO is that true for functions?
            Ok(SourceExpr { expr, span })
        }
        Variable(name) => Err(Error::with_span(
            format!("Variable '{}' not found in context", name),
            span,
        )),
        Seq { offset, waveform } => {
            let offset = evaluate_closed(*offset)?;
            let waveform = evaluate_closed(*waveform)?;
            Ok(SourceExpr::from(Seq {
                offset: Box::new(offset),
                waveform: Box::new(waveform),
            }))
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
        Application { function, argument } => {
            let function = evaluate_closed(*function)?;
            let argument = evaluate_closed(*argument)?;
            match (function.expr, argument) {
                (Function { pattern, body }, argument) => {
                    let mut context = Vec::new();
                    extend_context(&mut context, &pattern, &argument)?;
                    let body = substitute(&context, *body);
                    evaluate_closed(body)
                }
                (BuiltIn { function, .. }, argument) => {
                    // Builtins operate on bare `Expr<M>` values, so unwrap
                    // the SourceExpr children before calling and wrap the
                    // result. Use the outer Application's span (`span`) so
                    // errors point at the whole `f(x, y)` call site — builtins
                    // themselves don't see spans.
                    let actuals: Vec<Expr<M, S>> = match argument.expr {
                        Tuple(actuals) => actuals.into_iter().map(|s| s.expr).collect(),
                        other => vec![other],
                    };
                    let result = function.0(actuals);
                    match result {
                        Expr::Error(s) => Err(Error::with_span(s, span.clone())),
                        _ => Ok(SourceExpr::from(result)),
                    }
                }
                (function, actuals) => Err(Error::with_span(
                    format!("Invalid application: {} {}", function, actuals),
                    span.clone(),
                )),
            }
        }
        Tuple(exprs) => Ok(SourceExpr::from(Tuple(
            exprs
                .into_iter()
                .map(|e| evaluate_closed(e))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        List(exprs) => Ok(SourceExpr::from(List(
            exprs
                .into_iter()
                .map(|e| evaluate_closed(e))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
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

/// Walks `bindings`, accumulating evaluated entries into `context`. `Open`
/// bindings recurse through `resolve` to pull in their referenced module's
/// bindings.
///
fn build_context<'a, M, S, F>(
    resolve: &F,
    bindings: &'a [SourceBinding<M, S>],
    context: &mut Vec<(String, SourceExpr<M, S>)>,
) -> Result<(), Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug,
    S: Clone + Debug,
{
    for source_binding in bindings {
        match &source_binding.binding {
            Binding::Open(path) => {
                let module = resolve(path)?;
                build_context(resolve, module, context)?;
            }
            Binding::Definition(pattern, def_expr) => {
                let substituted = substitute(context, def_expr.clone());
                let value = evaluate_closed(substituted)?;
                extend_context(context, pattern, &value)?;
            }
            // Empty bindings have no semantic content — they exist only to
            // anchor annotation/comment spans for source preservation.
            Binding::Empty => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    #[test]
    fn test_function_eval() {
        let resolve = |_: &[String]| Err(Error::new("no bindings".to_string()));

        let input = "(fn(x) => fn(x) => x)(7)(5)";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        println!("Parsed expression: {}", expr);
        let evaluated = evaluate(resolve, &[], expr).unwrap();
        assert_eq!(format!("{}", evaluated), "5");

        let input = "(fn(x) => fn(y, z) => (x, y, z))(3)(4, 5)";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        let evaluated = evaluate(resolve, &[], expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");

        let input = "(fn(x, (y, z)) => (x, y, z))(3, (4, 5))";
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        let evaluated = evaluate(resolve, &[], expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");
    }
}
