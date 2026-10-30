//! Evaluation of tuun expressions: builds a context from bindings (resolving
//! `open`s through a caller-supplied module resolver), substitutes it into an
//! expression, and reduces the result to a value.

use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Display};

use std::sync::Arc;

use crate::expr::{
    Binding, BuiltInFn, Error, Expr, NamedExpr, NamedExprs, Pattern, SourceBinding, SourceExpr,
    Span,
};
use crate::trace::{self, CarriesTrace, Step, Trace, Via};
use crate::waveform::Waveform;

/// A context entry: a bound name and the closed value it evaluates to.
pub type ContextEntry<M, S> = NamedExpr<M, S>;

/// Identifies a module by the slice of bindings it was resolved to: its
/// address and length.
type ModuleKey = (usize, usize);

/// Returns the key identifying `module`.
fn module_key<M, S>(module: &[SourceBinding<M, S>]) -> ModuleKey {
    (module.as_ptr() as usize, module.len())
}

/// The evaluated exports of modules, carried between evaluations so that an
/// unchanged module is evaluated once rather than once per evaluation.
///
/// An entry is reused only while the resolver still returns the same slice
/// (by address and length) for the module and for every module it opened or
/// used, transitively. So a resolver used with a cache must not free a slice
/// it has returned and reuse its address for different bindings.
///
/// A reused module is not evaluated again, so built-ins applied in its
/// definitions (such as `debug`) run only when it is first evaluated, and
/// again after it or one of its dependencies resolves to a new slice.
pub struct ModuleCache<M, S> {
    entries: HashMap<Vec<String>, CachedModule<M, S>>,
}

impl<M, S> Default for ModuleCache<M, S> {
    fn default() -> ModuleCache<M, S> {
        ModuleCache {
            entries: HashMap::new(),
        }
    }
}

/// One module's exports and the slices they were built from.
struct CachedModule<M, S> {
    key: ModuleKey,
    /// The path and slice of each module resolved while building `exports`.
    dependencies: Vec<(Vec<String>, ModuleKey)>,
    exports: Vec<ContextEntry<M, S>>,
}

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
/// `context` should be closed values. On success, the resulting expression will
/// be closed.
///
/// Fails on expressions which are open with respect to context.
///
/// Rebuilt nodes keep their spans and vias as provenance. A substituted
/// variable keeps the span of the value's defining expression, and records
/// where the variable was written as a `UseSite` step on the value's via and
/// on the vias of its surface parts (see [`push_step`]).
fn substitute<M, S>(
    context: &mut Vec<(String, SourceExpr<M, S>)>,
    expr: SourceExpr<M, S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    use Expr::{Application, Bool, BuiltIn, Function, IfThenElse, List, String, Tuple, Variable};
    let SourceExpr { expr, span, via } = expr;
    Ok(match expr {
        Bool(_) | String(_) => SourceExpr { expr, span, via },
        Expr::Waveform(w) => SourceExpr {
            expr: Expr::Waveform(w),
            span,
            via,
        },
        Expr::Seq { offset, waveform } => SourceExpr {
            expr: Expr::Seq {
                offset: Box::new(substitute(context, *offset)?),
                waveform: Box::new(substitute(context, *waveform)?),
            },
            span,
            via,
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
                .map(|(name, value)| Ok((name, substitute(context, value)?)))
                .collect::<Result<_, self::Error<S>>>()?;
            let len = context.len();
            for param in &positional {
                extend_with_trivial_context(context, param);
            }
            for (name, _) in &named {
                context.push((name.clone(), SourceExpr::variable(name.clone())));
            }
            let body = substitute(context, *body);
            context.truncate(len);
            let body = body?;
            SourceExpr {
                expr: Expr::Function {
                    positional,
                    named,
                    body: Box::new(body),
                },
                span,
                via,
            }
        }
        BuiltIn { name, function } => SourceExpr {
            expr: Expr::BuiltIn { name, function },
            span,
            via,
        },
        // A bound module's entries are closed values (see `BoundModule`), so
        // there is nothing to substitute into.
        Expr::BoundModule(entries) => SourceExpr {
            expr: Expr::BoundModule(entries),
            span,
            via,
        },
        Expr::Project { module, name } => {
            // Projecting from a variable bound to a module copies only the
            // projected entry, not the whole module, recording the use site
            // as evaluating the projection would.
            if let Variable(module_name) = &module.expr
                && let Some(Expr::BoundModule(entries)) =
                    lookup(context, module_name).map(|value| &value.expr)
                && let Some((_, value)) = entries.iter().find(|(n, _)| *n == name)
            {
                return Ok(match span {
                    Some(span) => push_step(value.clone(), &Step::UseSite(span)),
                    None => value.clone(),
                });
            }
            SourceExpr {
                expr: Expr::Project {
                    module: Box::new(substitute(context, *module)?),
                    name,
                },
                span,
                via,
            }
        }
        Variable(name) => {
            let Some(value) = lookup(context, &name) else {
                return Err(self::Error::eval(
                    format!("Variable '{}' not found in context", name),
                    span,
                ));
            };
            // A parameter bound to itself keeps this node, and with it the
            // place the variable was written.
            if matches!(&value.expr, Variable(bound) if *bound == name) {
                return Ok(SourceExpr {
                    expr: Variable(name),
                    span,
                    via,
                });
            }
            return Ok(match span {
                Some(span) => push_step(value.clone(), &Step::UseSite(span)),
                None => value.clone(),
            });
        }
        IfThenElse {
            condition,
            then,
            else_,
        } => {
            let condition = Box::new(substitute(context, *condition)?);
            let then = Box::new(substitute(context, *then)?);
            let else_ = Box::new(substitute(context, *else_)?);
            SourceExpr {
                expr: Expr::IfThenElse {
                    condition,
                    then,
                    else_,
                },
                span,
                via,
            }
        }
        Application {
            function,
            positional,
            named,
        } => {
            let function = substitute(context, *function)?;
            let positional = positional
                .into_iter()
                .map(|a| substitute(context, a))
                .collect::<Result<_, _>>()?;
            let named = named
                .into_iter()
                .map(|(name, value)| Ok((name, substitute(context, value)?)))
                .collect::<Result<_, self::Error<S>>>()?;
            SourceExpr {
                expr: Expr::Application {
                    function: Box::new(function),
                    positional,
                    named,
                },
                span,
                via,
            }
        }
        Tuple(exprs) => SourceExpr {
            expr: Expr::Tuple(
                exprs
                    .into_iter()
                    .map(|e| substitute(context, e))
                    .collect::<Result<_, _>>()?,
            ),
            span,
            via,
        },
        List(exprs) => SourceExpr {
            expr: Expr::List(
                exprs
                    .into_iter()
                    .map(|e| substitute(context, e))
                    .collect::<Result<_, _>>()?,
            ),
            span,
            via,
        },
        error @ Expr::Error(_) => SourceExpr {
            expr: error,
            span,
            via,
        },
    })
}

/// Returns the value `context` binds `name` to: its latest entry for `name`.
fn lookup<'c, M, S>(
    context: &'c [(String, SourceExpr<M, S>)],
    name: &str,
) -> Option<&'c SourceExpr<M, S>> {
    context
        .iter()
        .rev()
        .find(|(bound, _)| bound == name)
        .map(|(_, value)| value)
}

fn extend_context<M, S>(
    context: &mut Vec<(String, SourceExpr<M, S>)>,
    pattern: &Pattern,
    argument: SourceExpr<M, S>,
) -> Result<(), Error<S>>
where
    M: Clone + Debug + Display,
    S: Clone + Debug,
{
    match (pattern, argument.expr) {
        (Pattern::Identifier(name), expr) => {
            context.push((name.clone(), SourceExpr { expr, ..argument }));
            Ok(())
        }
        (Pattern::Tuple(patterns), Expr::Tuple(arguments)) => {
            if patterns.len() != arguments.len() {
                return Err(Error::eval(
                    format!(
                        "Mismatched number of elements in pattern {} and arguments {}",
                        pattern,
                        Expr::Tuple(arguments)
                    ),
                    argument.span.clone(),
                ));
            }
            for (pattern, argument) in patterns.iter().zip(arguments) {
                extend_context(context, pattern, argument)?;
            }
            Ok(())
        }
        (_, expr) => Err(Error::eval(
            format!(
                "Pattern {} does not match actual expression {}",
                pattern, expr
            ),
            argument.span.clone(),
        )),
    }
}

/// Returns where an application at `span` of the function expression
/// `function` (not yet evaluated) was written as a call: the application
/// itself, or for a pipe `a | f(…)`, whose function expression starts after
/// the application does, the function expression `f(…)`.
///
/// Returns `None` when the application has no span, or when its function
/// expression wasn't written at all, as in the function a `let` desugars to.
fn call_site<M, S: Clone>(span: &Option<Span<S>>, function: &SourceExpr<M, S>) -> Option<Span<S>> {
    let span = span.as_ref()?;
    // Substitution replaces a variable with its value and records where the
    // variable was written as the newest step of its `via`; any other
    // function expression is still the node the parser wrote, with no `via`.
    let written = match function.via.as_deref() {
        Some(Via {
            step: Step::UseSite(written),
            ..
        }) => written,
        _ => function.span.as_ref()?,
    };
    // TODO a parenthesized function expression, as in `(fn (x) => x)(a)`,
    // also starts after its application does, so that call is sited at
    // `fn (x) => x` as if it were a pipe. The function and the argument
    // still light, but the whole call doesn't.
    Some(if written.range.start > span.range.start {
        written.clone()
    } else {
        span.clone()
    })
}

/// A built-in application as the built-in sees it: where the call was
/// written, and the traces of its function expression and its arguments.
// TODO is this the right place for this? maybe trace.rs?
pub struct BuiltInCall<S> {
    span: Option<Span<S>>,
    head: Arc<Trace<S>>,
    parents: Vec<Arc<Trace<S>>>,
}

impl<S: Clone> BuiltInCall<S> {
    /// Returns the call at `span` of a function expression whose trace is
    /// `head` to `arguments`.
    pub fn new<M: CarriesTrace<S>>(
        span: Option<Span<S>>,
        head: Trace<S>,
        arguments: &[SourceExpr<M, S>],
    ) -> BuiltInCall<S> {
        BuiltInCall {
            span,
            head: Arc::new(head),
            parents: arguments
                .iter()
                .map(|argument| Arc::new(Trace::of(argument)))
                .collect(),
        }
    }

    /// Returns `expr` as a new value this call computed, one whose audible
    /// samples it decides: written at the call, with a
    /// [`Step::Computed`] step.
    pub fn computed<M: CarriesTrace<S>>(&self, expr: Expr<M, S>) -> SourceExpr<M, S> {
        let step = Step::Computed {
            head: self.head.clone(),
            parents: self.parents.clone(),
        };
        self.written(expr, step)
    }

    /// Returns `expr` as a new value this call assembled, one whose audible
    /// samples, if any, are its arguments': written at the call, with a
    /// [`Step::Assembled`] step.
    pub fn assembled<M: CarriesTrace<S>>(&self, expr: Expr<M, S>) -> SourceExpr<M, S> {
        self.written(expr, Step::Assembled)
    }

    fn written<M: CarriesTrace<S>>(&self, expr: Expr<M, S>, step: Step<S>) -> SourceExpr<M, S> {
        SourceExpr {
            expr,
            span: self.span.clone(),
            via: Some(Via::push::<M>(step, None)),
        }
    }
}

/// Returns the value of `argument`, with each waveform at its surface marked
/// with the argument's trace (or a part's own), ready to embed in a node a
/// built-in builds.
///
/// See [`tag_in`] for which waveforms are marked.
pub fn embed<M: CarriesTrace<S>, S: Clone>(argument: SourceExpr<M, S>) -> Expr<M, S> {
    let trace = Arc::new(Trace::of(&argument));
    tag_in(argument.expr, &trace)
}

/// Applies the built-in `builtin` to `arguments` in a call at `span` whose
/// function expression has the trace `head`.
fn apply_builtin<M, S>(
    builtin: &BuiltInFn<M, S>,
    arguments: Vec<SourceExpr<M, S>>,
    span: Option<Span<S>>,
    head: Trace<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let call = BuiltInCall::new(span, head, &arguments);
    // Errors point at the whole `f(x, y)` call site, since built-ins don't
    // report spans. An error that already has a span came from a function the
    // built-in applied (e.g. `map`'s), and that span is more precise, so keep
    // it.
    let result = builtin.0(arguments, &call).map_err(|error| error.or_at(call.span.clone()))?;
    // A sub-expression with neither span nor via lights nothing: a built result
    // or sub-expression should come from `computed` or `assembled`. A mark at
    // the surface would be an embedded argument handed back, whose trace is the
    // sub-expressions's own.
    debug_assert!(
        surface_is_traced(&result),
        "a built-in's result has a sub-expression with no span or via, or a trace mark at its surface"
    );
    Ok(result)
}

/// Returns whether `value` and each sub-expression at its surface (`Seq` parts,
/// list elements, tuple fields, and theirs) has a span or a via, and no
/// waveform there has a trace mark at its root.
fn surface_is_traced<M: CarriesTrace<S>, S>(value: &SourceExpr<M, S>) -> bool {
    if value.span.is_none() && value.via.is_none() {
        return false;
    }
    match &value.expr {
        Expr::Waveform(w) => trace::root_trace(w).is_none(),
        Expr::Seq { offset, waveform } => surface_is_traced(offset) && surface_is_traced(waveform),
        Expr::List(exprs) | Expr::Tuple(exprs) => exprs.iter().all(surface_is_traced),
        _ => true,
    }
}

/// Returns `expr` with each waveform at its surface (the root, `Seq` parts,
/// list elements, tuple fields) marked with the trace of what holds it, the
/// container's `trace` or the part's own, for [`embed`].
///
/// A waveform's old trace marks at its root are replaced, never stacked. A
/// trace that doesn't reach program text could never light anything, so it
/// marks nothing.
fn tag_in<M: CarriesTrace<S>, S: Clone>(expr: Expr<M, S>, trace: &Arc<Trace<S>>) -> Expr<M, S> {
    match expr {
        Expr::Waveform(w) => {
            let w = trace::strip_root_trace_marks(w);
            let id = match trace.reaches_program && trace::trace_marking() {
                true => M::from_trace(trace),
                false => None,
            };
            Expr::Waveform(match id {
                Some(id) => Waveform::Marked {
                    id,
                    waveform: Box::new(w),
                },
                None => w,
            })
        }
        Expr::Seq { offset, waveform } => Expr::Seq {
            offset: Box::new(tag_part(*offset, trace)),
            waveform: Box::new(tag_part(*waveform, trace)),
        },
        Expr::List(exprs) => Expr::List(exprs.into_iter().map(|e| tag_part(e, trace)).collect()),
        Expr::Tuple(exprs) => Expr::Tuple(exprs.into_iter().map(|e| tag_part(e, trace)).collect()),
        expr => expr,
    }
}

/// Returns the surface part `part` of a value whose trace is `container`,
/// tagged as [`tag_in`] does: with the container's trace when the part has no
/// span or via of its own, else with its own.
fn tag_part<M: CarriesTrace<S>, S: Clone>(
    part: SourceExpr<M, S>,
    container: &Arc<Trace<S>>,
) -> SourceExpr<M, S> {
    let SourceExpr { expr, span, via } = part;
    let expr = match expr {
        Expr::Waveform(_) | Expr::Seq { .. } | Expr::List(_) | Expr::Tuple(_) => {
            if span.is_none() && via.is_none() {
                tag_in(expr, container)
            } else {
                let own = Trace::new::<M>(span.clone(), via.clone());
                tag_in(expr, &Arc::new(own))
            }
        }
        expr => expr,
    };
    SourceExpr { expr, span, via }
}

/// Returns `value` with `step` added as the newest step of its via, and of the
/// via of each part at its surface (`Seq` parts, list elements, tuple fields,
/// and theirs) that has a span or via of its own.
///
/// An anonymous part takes its container's trace when it is tagged (see
/// [`tag_part`]), so it sees the step there.
fn push_step<M: CarriesTrace<S>, S: Clone>(
    value: SourceExpr<M, S>,
    step: &Step<S>,
) -> SourceExpr<M, S> {
    let SourceExpr { expr, span, via } = value;
    let via = Some(Via::push::<M>(step.clone(), via));
    SourceExpr {
        expr: push_step_into_parts(expr, step),
        span,
        via,
    }
}

/// Returns `expr` with `step` pushed, as [`push_step`] does, onto each part
/// at its surface that has a span or via of its own.
fn push_step_into_parts<M: CarriesTrace<S>, S: Clone>(
    expr: Expr<M, S>,
    step: &Step<S>,
) -> Expr<M, S> {
    let push = |part: SourceExpr<M, S>| {
        if part.span.is_none() && part.via.is_none() {
            SourceExpr {
                expr: push_step_into_parts(part.expr, step),
                ..part
            }
        } else {
            push_step(part, step)
        }
    };
    match expr {
        Expr::Seq { offset, waveform } => Expr::Seq {
            offset: Box::new(push(*offset)),
            waveform: Box::new(push(*waveform)),
        },
        Expr::List(exprs) => Expr::List(exprs.into_iter().map(push).collect()),
        Expr::Tuple(exprs) => Expr::Tuple(exprs.into_iter().map(push).collect()),
        expr => expr,
    }
}

/// Evaluates a closed expression to a value. Closed expressions do not contain
/// variables.
///
/// Equivalent to [`evaluate`] with no bindings and a resolver that always
/// fails, but skips the needless substitution pass.
///
/// The result's spans are provenance (see [`SourceExpr`]): an expression that
/// is already a value keeps its span, an extracted sub-value (a chosen branch,
/// an applied function's body) keeps its own, and a built-in application's
/// result carries the span of the call site, unless it is an argument (or a
/// part of one) the built-in handed back.
pub fn evaluate_closed<M, S>(expr: SourceExpr<M, S>) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + fmt::Display + fmt::Debug + CarriesTrace<S>,
    S: Clone + fmt::Debug,
{
    use Expr::{
        Application, Bool, BuiltIn, Function, IfThenElse, List, Seq, String, Tuple, Variable,
        Waveform,
    };
    let SourceExpr { expr, span, via } = expr;
    match expr {
        Bool(_) | String(_) | Waveform(_) => {
            // For values, we can preserve the input span, since it still faithfully
            // represents the value.
            Ok(SourceExpr { expr, span, via })
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
                via,
            })
        }
        Variable(name) => Err(Error::eval(
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
                via,
            })
        }
        BuiltIn { name, function } => Ok(SourceExpr {
            expr: BuiltIn { name, function },
            span,
            via,
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
                _ => Err(Error::eval(
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
            let site = call_site(&span, &function);
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
                            return Err(Error::eval(
                                format!("named parameter \"{}\" appears more than once", name),
                                span.clone(),
                            ));
                        }
                        if !defaults.iter().any(|(n, _)| n == name) {
                            return Err(Error::eval(
                                format!("no named parameter \"{}\"", name),
                                span.clone(),
                            ));
                        }
                    }
                    if pos_args.len() > pos_params.len() {
                        return Err(Error::eval(
                            "extra positional parameter".to_string(),
                            span.clone(),
                        ));
                    }
                    if pos_args.len() < pos_params.len() {
                        return Err(Error::eval(
                            format!("missing parameter \"{}\"", pos_params[pos_args.len()]),
                            span.clone(),
                        ));
                    }
                    let mut context = Vec::new();
                    for (param, argument) in pos_params.iter().zip(pos_args) {
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
                    let body = substitute(&mut context, *body)?;
                    let result = evaluate_closed(body)?;
                    Ok(match site {
                        Some(site) => push_step(result, &Step::Call(site)),
                        None => result,
                    })
                }
                (
                    BuiltIn {
                        name,
                        function: builtin,
                    },
                    arguments,
                ) => {
                    if let Some((named_name, _)) = named.first() {
                        return Err(Error::eval(
                            format!(
                                "named argument \"{}\" is not supported by built-in \"{}\"",
                                named_name, name
                            ),
                            span.clone(),
                        ));
                    }
                    let head = Trace::new::<M>(function.span, function.via);
                    apply_builtin(&builtin, arguments, span, head)
                }
                (function, _) => Err(Error::internal(
                    format!("Invalid application: {}", function),
                    span.clone(),
                )),
            }
        }
        Expr::BoundModule(entries) => Ok(SourceExpr {
            expr: Expr::BoundModule(entries),
            span,
            via,
        }),
        Expr::Project { module, name } => match evaluate_closed(*module)?.expr {
            Expr::BoundModule(entries) => match entries.into_iter().find(|(n, _)| n == &name) {
                Some((_, value)) => Ok(match span {
                    Some(span) => push_step(value, &Step::UseSite(span)),
                    None => value,
                }),
                None => Err(Error::eval(
                    format!("Module has no binding '{}'", name),
                    span,
                )),
            },
            other => Err(Error::eval(
                format!("Cannot project '{}' from a non-module: {}", name, other),
                span,
            )),
        },
        Tuple(exprs) => Ok(SourceExpr {
            expr: Tuple(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            span,
            via,
        }),
        List(exprs) => Ok(SourceExpr {
            expr: List(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            span,
            via,
        }),
        // A parser error node, so the message is already the user's to read.
        Expr::Error(message) => Err(Error::eval(message, span)),
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
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    evaluate_cached(resolve, &mut ModuleCache::default(), bindings, expr)
}

/// Evaluates `bindings` and then `expr`, as [`evaluate`] does, reusing and
/// updating the module exports in `cache`.
pub fn evaluate_cached<'a, M, S, F>(
    resolve: F,
    cache: &mut ModuleCache<M, S>,
    bindings: &'a [SourceBinding<M, S>],
    expr: SourceExpr<M, S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let mut context = evaluate_bindings_cached(resolve, cache, bindings)?;
    evaluate_with_context(&mut context, expr)
}

/// Evaluates `bindings` and returns the resulting context: one entry per
/// bound name, in binding order, so a later entry shadows an earlier one
/// with the same name.
///
/// `resolve` loads `open`ed modules, as in [`evaluate`].
pub fn evaluate_bindings<'a, M, S, F>(
    resolve: F,
    bindings: &'a [SourceBinding<M, S>],
) -> Result<Vec<ContextEntry<M, S>>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    evaluate_bindings_cached(resolve, &mut ModuleCache::default(), bindings)
}

/// Evaluates `bindings` and returns the resulting context, as
/// [`evaluate_bindings`] does, reusing and updating the module exports in
/// `cache`.
pub fn evaluate_bindings_cached<'a, M, S, F>(
    resolve: F,
    cache: &mut ModuleCache<M, S>,
    bindings: &'a [SourceBinding<M, S>],
) -> Result<Vec<ContextEntry<M, S>>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let mut context = Vec::new();
    extend_context_cached(resolve, cache, bindings, &mut context)?;
    Ok(context)
}

/// Evaluates `bindings` after the entries already in `context`, appending an
/// entry to `context` for each bound name, and reusing and updating the
/// module exports in `cache`.
///
/// `context` should be a context [`evaluate_bindings`] or this function
/// returned. On failure, `context` holds the entries of the bindings before
/// the one that failed.
///
/// # Example
///
/// ```
/// use tuun::eval::{self, ModuleCache};
/// use tuun::expr::Error;
/// use tuun::parser::{parse_module, parse_program};
///
/// let resolve = |_: &[String]| Err(Error::eval_here("no modules".to_string()));
/// let mut cache = ModuleCache::default();
/// let (first, _) = parse_module::<u32, _>("a = 1;", ()).unwrap();
/// let (second, _) = parse_module::<u32, _>("b = a;", ()).unwrap();
/// let mut context = Vec::new();
/// eval::extend_context_cached(resolve, &mut cache, &first, &mut context).unwrap();
/// eval::extend_context_cached(resolve, &mut cache, &second, &mut context).unwrap();
/// let b = parse_program::<u32, _>("b", ()).unwrap();
/// let value = eval::evaluate_with_context(&mut context, b).unwrap();
/// assert_eq!(format!("{}", value), "1");
/// ```
pub fn extend_context_cached<'a, M, S, F>(
    resolve: F,
    cache: &mut ModuleCache<M, S>,
    bindings: &'a [SourceBinding<M, S>],
    context: &mut Vec<ContextEntry<M, S>>,
) -> Result<(), Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let mut modules = Modules {
        resolve: &resolve,
        cache,
        open: Vec::new(),
        resolved: Vec::new(),
    };
    build_context(&mut modules, bindings, context)?;
    Ok(())
}

/// Substitutes `context` into `expr` and reduces the result to a value.
///
/// `context` should be a context [`evaluate_bindings`] or
/// [`extend_context_cached`] returned; it holds the same entries afterwards.
pub fn evaluate_with_context<M, S>(
    context: &mut Vec<(String, SourceExpr<M, S>)>,
    expr: SourceExpr<M, S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let expr = substitute(context, expr)?;
    evaluate_closed(expr)
}

/// The modules one evaluation of bindings loads: how it resolves them, the
/// cache it reuses them from, and what it has loaded so far.
struct Modules<'r, 'c, F, M, S> {
    resolve: &'r F,
    cache: &'c mut ModuleCache<M, S>,
    /// The modules being opened, innermost last.
    open: Vec<ModuleKey>,
    /// The path and slice of each module resolved so far, in order.
    resolved: Vec<(Vec<String>, ModuleKey)>,
}

/// Returns what the module `module`, resolved from `path`, exports: from
/// `modules`' cache when nothing it was built from has changed, else by
/// evaluating it. Refuses a module that is already being opened.
///
/// The checker reports the same cycle first, so reaching this means the
/// evaluator was driven without it.
fn module_exports<'a, M, S, F>(
    modules: &mut Modules<'_, '_, F, M, S>,
    module: &'a [SourceBinding<M, S>],
    path: &[String],
    span: &Option<Span<S>>,
) -> Result<Vec<ContextEntry<M, S>>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let key = module_key(module);
    if modules.open.contains(&key) {
        return Err(Error::eval(
            format!("module '{}' is opened from itself", path.join(".")),
            span.clone(),
        ));
    }
    modules.resolved.push((path.to_vec(), key));
    if let Some(cached) = modules.cache.entries.get(path)
        && cached.key == key
        && cached.dependencies.iter().all(|(path, key)| {
            (modules.resolve)(path).is_ok_and(|module| module_key(module) == *key)
        })
    {
        modules.resolved.extend(cached.dependencies.iter().cloned());
        return Ok(cached.exports.clone());
    }
    let start = modules.resolved.len();
    modules.open.push(key);
    let mut module_context = Vec::new();
    let own = build_context(modules, module, &mut module_context);
    modules.open.pop();
    let mut own = own?.into_iter().peekable();
    let exports: Vec<_> = module_context
        .into_iter()
        .enumerate()
        .filter(|(i, _)| own.next_if_eq(i).is_some())
        .map(|(_, entry)| entry)
        .collect();
    let mut dependencies = modules.resolved[start..].to_vec();
    dependencies.sort();
    dependencies.dedup();
    modules.cache.entries.insert(
        path.to_vec(),
        CachedModule {
            key,
            dependencies,
            exports: exports.clone(),
        },
    );
    Ok(exports)
}

/// Evaluates `bindings`, appending an entry to `context` for each bound name.
/// Returns the indices into `context` of the entries `bindings` defines itself
/// (as opposed to those brought in by `open`).
fn build_context<'a, M, S, F>(
    modules: &mut Modules<'_, '_, F, M, S>,
    bindings: &'a [SourceBinding<M, S>],
    context: &mut Vec<ContextEntry<M, S>>,
) -> Result<Vec<usize>, Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let mut own = Vec::new();
    for source_binding in bindings {
        match &source_binding.binding {
            Binding::Open(path) => {
                let module = (modules.resolve)(path)?;
                let exports = module_exports(modules, module, path, &source_binding.span)?;
                context.extend(exports);
            }
            Binding::Use(path) => {
                let Some(name) = path.last() else {
                    return Err(Error::eval(
                        "`use` requires a module path".to_string(),
                        source_binding.span.clone(),
                    ));
                };
                let module = (modules.resolve)(path)?;
                let exports = module_exports(modules, module, path, &source_binding.span)?;
                context.push((
                    name.clone(),
                    SourceExpr::from(Expr::BoundModule(dedup_last_wins(exports))),
                ));
            }
            Binding::Definition(pattern, def_expr) => {
                let substituted = substitute(context, def_expr.clone())?;
                let value = evaluate_closed(substituted)?;
                let before = context.len();
                extend_context(context, pattern, value)?;
                own.extend(before..context.len());
            }
            // Empty bindings have no semantic content — they exist only to
            // anchor annotation/comment spans for source preservation.
            Binding::Empty => {}
        }
    }
    Ok(own)
}

/// Returns `entries` with duplicate names collapsed so each name appears
/// once, bound to the value of its last occurrence.
fn dedup_last_wins<M, S>(entries: Vec<ContextEntry<M, S>>) -> Vec<ContextEntry<M, S>> {
    let mut out: Vec<ContextEntry<M, S>> = Vec::new();
    for (name, value) in entries {
        match out.iter_mut().find(|(n, _)| *n == name) {
            Some(existing) => existing.1 = value,
            None => out.push((name, value)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins;
    use crate::expr::ErrorKind;
    use crate::parser::{parse_module, parse_program};

    /// Parses and evaluates `input` with the built-ins in scope.
    fn eval_with_builtins(input: &str) -> Result<SourceExpr<u32, ()>, Error<()>> {
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        evaluate(
            |_: &[String]| Err(Error::eval_here("no modules".to_string())),
            &bindings,
            expr,
        )
    }

    #[test]
    fn test_sequence_requires_seqs() {
        let seq = "time | fin(time - 1) | seq(time - 1)";
        // A fold of seqs is a seq; a singleton returns its element.
        assert!(matches!(
            eval_with_builtins(&format!("<[{seq}, {seq}]>"))
                .unwrap()
                .expr,
            Expr::Seq { .. }
        ));
        assert!(matches!(
            eval_with_builtins(&format!("<[{seq}]>")).unwrap().expr,
            Expr::Seq { .. }
        ));
        // A non-seq element is an error.
        assert!(eval_with_builtins(&format!("<[{seq}, time]>")).is_err());
        // The empty sequence is the empty seq — usable on `\`'s left.
        assert!(matches!(
            eval_with_builtins("<[]>").unwrap().expr,
            Expr::Seq { .. }
        ));
        assert!(matches!(
            eval_with_builtins("<[]> \\ time").unwrap().expr,
            Expr::Waveform(_)
        ));
    }

    #[test]
    fn test_merge_of_constants_folds() {
        // `&` on two constants is their sum: both are endless, so the merge
        // is point-wise addition everywhere and the shorter-operand extension
        // never applies.
        assert_eq!(format!("{}", eval_with_builtins("1 & 2").unwrap()), "3");
        // A constant merged with a non-constant stays a merge tree.
        assert_eq!(
            format!("{}", eval_with_builtins("1 & time").unwrap()),
            format!("{}", eval_with_builtins("(fn(w) => 1 & w)(time)").unwrap())
        );
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
        let error = eval_with_builtins("log(2, 8, y = 1)").unwrap_err();
        assert!(error.message().contains("built-in \"log\""), "{}", error);
    }

    #[test]
    fn test_named_defaults_evaluate_once() {
        use std::sync::{Arc, Mutex};

        let printed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&printed);
        let mut bindings: Vec<SourceBinding<u32, ()>> = Vec::new();
        builtins::add_bindings(&mut bindings);
        bindings.push(
            Binding::Definition(
                Pattern::Identifier("debug".to_string()),
                builtins::debug(move |line| sink.lock().unwrap().push(line.to_string())),
            )
            .into(),
        );
        let resolve = |_: &[String]| Err(Error::eval_here("no modules".to_string()));

        // The default is evaluated once, when the function value is
        // created — not at each of the three calls.
        let expr = parse_program::<u32, _>(
            "let f = fn(x, y = 1 | debug(\"once\")) => x, _ = f(1), _ = f(2) in f(3)",
            (),
        )
        .unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "3");
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] once: 1"]);

        // Even a function that is never applied evaluates its defaults.
        printed.lock().unwrap().clear();
        let expr = parse_program::<u32, _>("let f = fn(x, y = 1 | debug(\"once\")) => x in 0", ())
            .unwrap();
        evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] once: 1"]);
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
                Err(Error::eval_here(format!("no module {:?}", path)))
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
    fn a_cyclic_open_is_refused_not_recursed() {
        // The checker reports this first, so reaching the evaluator means
        // it was driven without one.
        let (a, _) = parse_module::<u32, _>("open b;\nx = 1;\n", ()).unwrap();
        let (b, _) = parse_module::<u32, _>("open a;\ny = 2;\n", ()).unwrap();
        let resolve = |path: &[String]| match path.join(".").as_str() {
            "a" => Ok(&a[..]),
            "b" => Ok(&b[..]),
            other => Err(Error::eval_here(format!("no module {}", other))),
        };
        let expr = parse_program::<u32, _>("0", ()).unwrap();
        let error = evaluate(resolve, &a, expr).unwrap_err();
        assert!(
            error.message().contains("is opened from itself"),
            "expected a cycle report, got {}",
            error.message()
        );
    }

    #[test]
    fn test_map_propagates_element_errors() {
        // A failing element application fails the whole `map`; the error must
        // not be demoted to an Error value inside the result list, where it
        // would flow onward as ordinary data. (Applying the float 1 can only
        // fail at application time, past substitution's unbound-name check.)
        let error = eval_with_builtins("map(fn(x) => x(1), [1, 2])").unwrap_err();
        // Applying a non-function is a error the checker is meant to rule
        // out, so it is reported as one.
        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(error.message(), "internal error: Invalid application: 1");
    }

    #[test]
    fn test_higher_order_builtins_keep_element_errors() {
        // An out-of-range `nth` inside an applied function is the caller's
        // error, not an internal one, and points at the `nth` call rather
        // than the enclosing `map`/`reduce`/`unfold`.
        for input in [
            "map(fn(i) => nth(i, [1, 2]), [0, 5])",
            "reduce(fn(a, i) => a + nth(i, [1, 2]), 0, [0, 5])",
            "unfold(fn(i) => nth(i, [1, 2]), 5, 2)",
        ] {
            let error = eval_with_builtins(input).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Eval, "{}", input);
            assert_eq!(error.message(), "no element with index 5", "{}", input);
            let start = input.find("nth").unwrap();
            let end = start + "nth(i, [1, 2])".len();
            assert_eq!(error.range(), Some(start..end), "{}", input);
        }
    }

    #[test]
    fn evaluating_in_a_context_leaves_its_entries() {
        let (bindings, errors) = parse_module::<u32, _>("a = 1; b = fn(x) => x;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |_: &[String]| Err(Error::eval_here("no modules".to_string()));
        let mut context = evaluate_bindings(resolve, &bindings).unwrap();
        let names = |context: &[ContextEntry<u32, ()>]| -> Vec<String> {
            context.iter().map(|(name, _)| name.clone()).collect()
        };
        let before = names(&context);
        for (input, succeeds) in [
            ("b(fn(y, z = a) => (y, z))", true),
            ("fn(y) => fn(z) => nope", false),
        ] {
            let expr = parse_program::<u32, _>(input, ()).unwrap();
            assert_eq!(evaluate_with_context(&mut context, expr).is_ok(), succeeds);
            assert_eq!(names(&context), before, "{}", input);
        }
    }

    #[test]
    fn test_unbound_variable_fails_at_definition() {
        // The function is never applied: the unbound name in its body still
        // fails when the definition itself is evaluated, with the span of
        // the unbound reference.
        let input = "let f = fn(x) => nope(x) in 1";
        let error = eval_with_builtins(input).unwrap_err();
        assert_eq!(error.message(), "Variable 'nope' not found in context");
        let start = input.find("nope").unwrap();
        assert_eq!(error.range(), Some(start..start + 4));

        // Same for a binding: the context build fails at the definition.
        let (bindings, errors) = parse_module::<u32, _>("f = fn(x) => nope(x);", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |_: &[String]| Err(Error::eval_here("no modules".to_string()));
        let error = evaluate_bindings(resolve, &bindings).unwrap_err();
        assert_eq!(error.message(), "Variable 'nope' not found in context");
    }

    #[test]
    fn test_use_binds_module_value() {
        let (b, errors) = parse_module::<u32, _>("two = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["b"] || path == ["nested", "b"] {
                Ok(&b[..])
            } else {
                Err(Error::eval_here(format!("no module {:?}", path)))
            }
        };

        let (bindings, errors) = parse_module::<u32, _>("use b;", ()).unwrap();
        assert!(errors.is_empty());
        let expr = parse_program::<u32, _>("b.two", ()).unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "2");

        // A dotted path binds its last component.
        let (bindings, errors) = parse_module::<u32, _>("use nested.b;", ()).unwrap();
        assert!(errors.is_empty());
        let expr = parse_program::<u32, _>("b.two", ()).unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "2");
    }

    #[test]
    fn test_use_projection_errors() {
        let (b, errors) = parse_module::<u32, _>("two = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::eval_here(format!("no module {:?}", path)))
            }
        };
        let (bindings, errors) = parse_module::<u32, _>("use b;", ()).unwrap();
        assert!(errors.is_empty());

        // Projecting a name the module doesn't bind.
        let expr = parse_program::<u32, _>("b.three", ()).unwrap();
        let error = evaluate(resolve, &bindings, expr).unwrap_err();
        assert_eq!(error.message(), "Module has no binding 'three'");

        // Projecting from a non-module value.
        let expr = parse_program::<u32, _>("let x = 1 in x.y", ()).unwrap();
        let error = evaluate(resolve, &bindings, expr).unwrap_err();
        assert_eq!(error.message(), "Cannot project 'y' from a non-module: 1");
    }

    #[test]
    fn test_use_module_last_definition_wins() {
        let (b, errors) = parse_module::<u32, _>("x = 1; x = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::eval_here(format!("no module {:?}", path)))
            }
        };
        let (bindings, errors) = parse_module::<u32, _>("use b;", ()).unwrap();
        assert!(errors.is_empty());

        let expr = parse_program::<u32, _>("b.x", ()).unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "2");

        // The bound value keeps one entry per name (observable via Display,
        // which lists the bound names).
        let context = evaluate_bindings(resolve, &bindings).unwrap();
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].0, "b");
        assert_eq!(format!("{}", context[0].1), "module {x}");
    }

    #[test]
    fn test_projection_substitutes_only_the_entry() {
        let (b, errors) = parse_module::<u32, _>("two = 2; three = 3;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::eval_here(format!("no module {:?}", path)))
            }
        };
        let (bindings, errors) = parse_module::<u32, _>("use b;", ()).unwrap();
        assert!(errors.is_empty());

        // The body of a function that is never applied holds the projected
        // value, not the module it was projected from.
        let expr = parse_program::<u32, _>("fn(x) => b.two", ()).unwrap();
        let Expr::Function { body, .. } = evaluate(resolve, &bindings, expr).unwrap().expr else {
            panic!("expected a function");
        };
        assert_eq!(format!("{}", body), "2");
    }

    #[test]
    fn test_uses_are_scoped() {
        let (b, errors) = parse_module::<u32, _>("two = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let (a, errors) = parse_module::<u32, _>("use b; alias = b.two;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["a"] {
                Ok(&a[..])
            } else if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::eval_here(format!("no module {:?}", path)))
            }
        };
        let (bindings, errors) = parse_module::<u32, _>("use a;", ()).unwrap();
        assert!(errors.is_empty());

        // `a`'s own definitions can project from the module `a` used...
        let expr = parse_program::<u32, _>("a.alias", ()).unwrap();
        let evaluated = evaluate(resolve, &bindings, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "2");

        // ...but using `a` does not re-export the module binding `b`...
        let expr = parse_program::<u32, _>("a.b", ()).unwrap();
        let error = evaluate(resolve, &bindings, expr).unwrap_err();
        assert_eq!(error.message(), "Module has no binding 'b'");

        // ...nor is `b` in scope directly.
        let expr = parse_program::<u32, _>("b.two", ()).unwrap();
        let error = evaluate(resolve, &bindings, expr).unwrap_err();
        assert_eq!(error.message(), "Variable 'b' not found in context");
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

    /// Tests of reusing module exports between evaluations, against a
    /// stand-in built-in that counts its applications.
    mod module_cache {
        use super::*;
        use std::cell::Cell;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Returns a module binding `tick`, which hands back its argument and
        /// counts each application in `count`.
        fn ticks(count: &Arc<AtomicUsize>) -> Vec<SourceBinding<u32, ()>> {
            let count = Arc::clone(count);
            vec![SourceBinding::definition(
                Pattern::Identifier("tick".to_string()),
                SourceExpr::from(Expr::BuiltIn {
                    name: "tick".to_string(),
                    function: BuiltInFn::new(move |mut arguments, _call| {
                        count.fetch_add(1, Ordering::SeqCst);
                        Ok(arguments.pop().unwrap())
                    }),
                }),
            )]
        }

        fn module(text: &str) -> Vec<SourceBinding<u32, ()>> {
            let (bindings, errors) = parse_module::<u32, _>(text, ()).unwrap();
            assert!(errors.is_empty());
            bindings
        }

        #[test]
        fn an_unchanged_module_is_evaluated_once() {
            let count = Arc::new(AtomicUsize::new(0));
            let ticks = ticks(&count);
            let a = module("open ticks; x = tick(1);");
            let resolve = |path: &[String]| match path.join(".").as_str() {
                "ticks" => Ok(&ticks[..]),
                "a" => Ok(&a[..]),
                other => Err(Error::eval_here(format!("no module {}", other))),
            };
            let bindings = module("open a;");
            let mut cache = ModuleCache::default();
            for _ in 0..3 {
                let expr = parse_program::<u32, _>("x", ()).unwrap();
                let value = evaluate_cached(resolve, &mut cache, &bindings, expr).unwrap();
                assert_eq!(format!("{}", value), "1");
            }
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn a_changed_dependency_reevaluates_its_dependents() {
            let count = Arc::new(AtomicUsize::new(0));
            let ticks = ticks(&count);
            let a = module("open ticks; open b; x = tick(y);");
            let b1 = module("y = 1;");
            let b2 = module("y = 2;");
            let b = Cell::new(&b1[..]);
            let resolve = |path: &[String]| match path.join(".").as_str() {
                "ticks" => Ok(&ticks[..]),
                "a" => Ok(&a[..]),
                "b" => Ok(b.get()),
                other => Err(Error::eval_here(format!("no module {}", other))),
            };
            let bindings = module("open a;");
            let mut cache = ModuleCache::default();
            let mut run = || {
                let expr = parse_program::<u32, _>("x", ()).unwrap();
                format!(
                    "{}",
                    evaluate_cached(resolve, &mut cache, &bindings, expr).unwrap()
                )
            };
            assert_eq!(run(), "1");
            // `b` resolves to a new slice, as a reloaded module does; `a`
            // still resolves to the same one.
            b.set(&b2[..]);
            assert_eq!(run(), "2");
            assert_eq!(run(), "2");
            assert_eq!(count.load(Ordering::SeqCst), 2);
        }
    }

    /// Tests of how evaluation records provenance, against stand-in
    /// built-ins and a stand-in mark id type that carries traces.
    mod provenance {
        use super::*;
        use crate::parser::{parse_module, parse_program};
        use crate::waveform::BinaryOperator;

        /// Which text a test span indexes.
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Text {
            Program,
            Library,
        }

        /// A mark id that carries a trace and nothing else.
        #[derive(Clone, Debug)]
        struct Traced(Arc<Trace<Text>>);

        impl Display for Traced {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "traced")
            }
        }

        impl CarriesTrace<Text> for Traced {
            fn from_trace(trace: &Arc<Trace<Text>>) -> Option<Self> {
                Some(Traced(trace.clone()))
            }
            fn as_trace(&self) -> Option<&Arc<Trace<Text>>> {
                Some(&self.0)
            }
            fn in_program(span: &Span<Text>) -> bool {
                span.source == Text::Program
            }
        }

        type Value = SourceExpr<Traced, Text>;

        type StandIn = fn(Vec<Value>, &BuiltInCall<Text>) -> Result<Value, Error<Text>>;

        fn waveform(expr: Expr<Traced, Text>) -> Waveform<Traced> {
            match expr {
                Expr::Waveform(w) => w,
                other => panic!("expected a waveform, got {}", other),
            }
        }

        /// The stand-in built-ins: `build` computes from both arguments,
        /// embedding them, `wrap` assembles one by embedding it, `pass` hands
        /// its argument back, `careless` builds a seq with a part it gives no
        /// trace, and `double` computes from a constant it consumes.
        fn stand_ins() -> Vec<SourceBinding<Traced, Text>> {
            fn def(name: &str, f: StandIn) -> SourceBinding<Traced, Text> {
                SourceBinding::definition(
                    Pattern::Identifier(name.to_string()),
                    SourceExpr::from(Expr::BuiltIn {
                        name: name.to_string(),
                        function: BuiltInFn::new(f),
                    }),
                )
            }
            vec![
                def("build", |mut arguments, call| {
                    let b = waveform(embed(arguments.pop().unwrap()));
                    let a = waveform(embed(arguments.pop().unwrap()));
                    Ok(call.computed(Expr::Waveform(Waveform::BinaryOp(
                        BinaryOperator::Add,
                        Box::new(a),
                        Box::new(b),
                    ))))
                }),
                def("wrap", |mut arguments, call| {
                    let a = waveform(embed(arguments.pop().unwrap()));
                    Ok(call.assembled(Expr::Waveform(Waveform::Fin {
                        length: Box::new(Waveform::Const(1.0)),
                        waveform: Box::new(a),
                        state: (),
                    })))
                }),
                def("pass", |mut arguments, _call| Ok(arguments.pop().unwrap())),
                def("careless", |mut arguments, call| {
                    Ok(call.assembled(Expr::Seq {
                        offset: Box::new(SourceExpr::float(1.0)),
                        waveform: Box::new(arguments.pop().unwrap()),
                    }))
                }),
                def("double", |arguments, call| {
                    let value = arguments[0].expr.as_const_float().expect("a constant");
                    Ok(call.computed(Expr::float(2.0 * value)))
                }),
            ]
        }

        const LIBRARY: &str = "f = fn(x) => x;\n\
                               g = fn(a) => fn(b) => b;\n\
                               c = build(1, 2);\n\
                               one = 1;\n\
                               passed = pass(one);\n";

        /// Evaluates `program` with the stand-ins and the library opened, and
        /// the library also bound as `lib`.
        fn traced(program: &str) -> Value {
            let prelude = stand_ins();
            let (mut library, errors) = parse_module(LIBRARY, Text::Library).unwrap();
            assert!(errors.is_empty());
            library.insert(0, Binding::Open(vec!["prelude".to_string()]).into());
            let resolve = |path: &[String]| match path.join(".").as_str() {
                "prelude" => Ok(&prelude[..]),
                "lib" => Ok(&library[..]),
                other => Err(Error::eval_here(format!("no module {}", other))),
            };
            let bindings: Vec<SourceBinding<Traced, Text>> = vec![
                Binding::Open(vec!["prelude".to_string()]).into(),
                Binding::Open(vec!["lib".to_string()]).into(),
                Binding::Use(vec!["lib".to_string()]).into(),
            ];
            let expr = parse_program(program, Text::Program).unwrap();
            evaluate(resolve, &bindings, expr).unwrap()
        }

        /// Returns the text `span` covers: its slice of `program`, or `lib`
        /// for library text.
        fn text(program: &str, span: &Span<Text>) -> String {
            match span.source {
                Text::Program => program[span.range.clone()].to_string(),
                Text::Library => "lib".to_string(),
            }
        }

        /// Returns where `trace` was last seen: its newest use site, else
        /// its span.
        fn seen(program: &str, trace: &Trace<Text>) -> String {
            match trace.via.as_deref() {
                Some(Via {
                    step: Step::UseSite(span),
                    ..
                }) => text(program, span),
                _ => trace
                    .span
                    .as_ref()
                    .map_or("-".to_string(), |s| text(program, s)),
            }
        }

        /// Returns the steps of `via`, oldest first: `use x`, `call f(1)`,
        /// `computed h` naming where the built-in's head was last seen, or
        /// `assembled`.
        fn steps(program: &str, via: &Option<Arc<Via<Text>>>) -> Vec<String> {
            let mut out = Vec::new();
            let mut via = via.as_deref();
            while let Some(step) = via {
                out.push(match &step.step {
                    Step::UseSite(span) => format!("use {}", text(program, span)),
                    Step::Call(span) => format!("call {}", text(program, span)),
                    Step::Computed { head, .. } => format!("computed {}", seen(program, head)),
                    Step::Assembled => "assembled".to_string(),
                });
                via = step.prev.as_deref();
            }
            out.reverse();
            out
        }

        /// Returns the `Computed` parents of `value`'s oldest step, each as
        /// where it was last seen.
        fn parents(program: &str, value: &Value) -> Vec<String> {
            let mut via = value.via.as_deref();
            while let Some(step) = via {
                if let Step::Computed { parents, .. } = &step.step {
                    return parents.iter().map(|p| seen(program, p)).collect();
                }
                via = step.prev.as_deref();
            }
            Vec::new()
        }

        /// Returns where each trace mark in `w` was written, in pre-order.
        fn marks(program: &str, w: &Waveform<Traced>) -> Vec<String> {
            match w {
                Waveform::Marked { id, waveform } => {
                    let mut out = vec![seen(program, &id.0)];
                    out.extend(marks(program, waveform));
                    out
                }
                Waveform::BinaryOp(_, a, b) => {
                    let mut out = marks(program, a);
                    out.extend(marks(program, b));
                    out
                }
                Waveform::Fin {
                    length, waveform, ..
                } => {
                    let mut out = marks(program, length);
                    out.extend(marks(program, waveform));
                    out
                }
                _ => Vec::new(),
            }
        }

        #[test]
        fn a_substituted_variable_records_its_use_site() {
            // `let` itself records nothing.
            let program = "let x = 1 in x";
            let value = traced(program);
            assert_eq!(text(program, value.span.as_ref().unwrap()), "1");
            assert_eq!(steps(program, &value.via), ["use x"]);
        }

        #[test]
        fn a_projection_records_its_use_site() {
            let program = "lib.one";
            let value = traced(program);
            assert_eq!(steps(program, &value.via), ["use lib.one"]);
        }

        #[test]
        fn a_user_call_records_its_site() {
            let program = "f(2)";
            let value = traced(program);
            assert_eq!(text(program, value.span.as_ref().unwrap()), "2");
            assert_eq!(steps(program, &value.via), ["use lib", "call f(2)"]);
        }

        #[test]
        fn use_sites_and_calls_reach_a_values_parts() {
            let program = "let x = [1] in f(x)";
            let Expr::List(parts) = traced(program).expr else {
                panic!("expected a list");
            };
            assert_eq!(text(program, parts[0].span.as_ref().unwrap()), "1");
            assert_eq!(
                steps(program, &parts[0].via),
                ["use x", "use lib", "call f(x)"]
            );
        }

        #[test]
        fn a_pipe_call_is_sited_at_its_function_expression() {
            let program = "2 | f";
            assert_eq!(steps(program, &traced(program).via), ["use lib", "call f"]);
            // A curried call: `g(3)` is a call of its own, and the pipe's
            // call is sited at it, not at the whole chain.
            let program = "2 | g(3)";
            assert_eq!(
                steps(program, &traced(program).via),
                ["use lib", "call g(3)"]
            );
        }

        #[test]
        fn a_routed_argument_keeps_its_trace() {
            let program = "let x = 1 in pass(x)";
            let value = traced(program);
            assert_eq!(text(program, value.span.as_ref().unwrap()), "1");
            assert_eq!(steps(program, &value.via), ["use x"]);
            assert!(marks(program, &waveform(value.expr)).is_empty());
        }

        #[test]
        fn routing_does_not_depend_on_trace_marks() {
            // An argument whose trace reaches no program text carries no mark.
            let program = "passed";
            assert_eq!(
                steps(program, &traced(program).via),
                ["use lib", "use passed"]
            );
            // Nor does any argument with trace marking off.
            trace::set_trace_marking(false);
            let program = "let x = 1 in pass(x)";
            let value = traced(program);
            trace::set_trace_marking(true);
            assert_eq!(steps(program, &value.via), ["use x"]);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic(expected = "a built-in's result has a sub-expression with no span or via")]
        fn a_built_part_with_no_trace_is_caught() {
            traced("careless(2)");
        }

        #[test]
        fn a_built_result_embeds_its_arguments_marks() {
            let program = "build(1, 2)";
            let value = traced(program);
            assert_eq!(steps(program, &value.via), ["computed build"]);
            assert_eq!(parents(program, &value), ["1", "2"]);
            assert_eq!(marks(program, &waveform(value.expr)), ["1", "2"]);
        }

        #[test]
        fn an_assembled_result_records_no_head_or_parents() {
            let program = "wrap(1)";
            let value = traced(program);
            assert_eq!(steps(program, &value.via), ["assembled"]);
            assert!(parents(program, &value).is_empty());
            assert_eq!(marks(program, &waveform(value.expr)), ["1"]);
        }

        #[test]
        fn a_consumed_argument_survives_only_in_parents() {
            // `double` reads its argument's value through the mark.
            let program = "double(3)";
            let value = traced(program);
            assert_eq!(steps(program, &value.via), ["computed double"]);
            assert_eq!(parents(program, &value), ["3"]);
            assert!(matches!(waveform(value.expr), Waveform::Const(v) if v == 6.0));
        }

        #[test]
        fn only_values_reaching_program_text_are_tagged() {
            // Built from library text only.
            let program = "c";
            assert!(marks(program, &waveform(traced(program).expr)).is_empty());
            // The same value reaches program text through its use site.
            let program = "build(c, 3)";
            assert_eq!(marks(program, &waveform(traced(program).expr)), ["c", "3"]);
        }

        #[test]
        fn trace_marking_off_makes_no_marks_but_still_records_via() {
            trace::set_trace_marking(false);
            let program = "build(1, 2)";
            let value = traced(program);
            trace::set_trace_marking(true);
            assert_eq!(steps(program, &value.via), ["computed build"]);
            assert!(marks(program, &waveform(value.expr)).is_empty());
        }

        /// A trace of program text written at `range`.
        fn written(range: std::ops::Range<usize>) -> Arc<Trace<Text>> {
            Arc::new(Trace::new::<Traced>(
                Some(Span::new(Text::Program, range)),
                None,
            ))
        }

        #[test]
        fn tagging_replaces_root_marks_rather_than_stacking() {
            let old = Waveform::Marked {
                id: Traced(written(0..1)),
                waveform: Box::new(Waveform::Const(1.0)),
            };
            let new = written(2..3);
            let Waveform::Marked { id, waveform } = waveform(tag_in(Expr::Waveform(old), &new))
            else {
                panic!("expected a marked node at the root");
            };
            assert!(Arc::ptr_eq(&id.0, &new));
            assert!(matches!(*waveform, Waveform::Const(v) if v == 1.0));
        }

        #[test]
        fn a_seq_part_with_its_own_trace_keeps_it() {
            let own = SourceExpr {
                expr: Expr::float(1.0),
                span: Some(Span::new(Text::Program, 0..1)),
                via: None,
            };
            let seq: Expr<Traced, Text> = Expr::Seq {
                offset: Box::new(SourceExpr::float(0.0)),
                waveform: Box::new(own),
            };
            let container = written(4..5);
            let Expr::Seq { offset, waveform } = tag_in(seq, &container) else {
                panic!("expected a seq");
            };
            // A part with no trace of its own takes its container's.
            let Expr::Waveform(Waveform::Marked { id, .. }) = &offset.expr else {
                panic!("expected a marked offset");
            };
            assert!(Arc::ptr_eq(&id.0, &container));
            // A part with its own keeps it.
            let Expr::Waveform(Waveform::Marked { id, .. }) = &waveform.expr else {
                panic!("expected a marked waveform");
            };
            assert_eq!(id.0.span.as_ref().unwrap().range, 0..1);
        }
    }
}
