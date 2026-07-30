//! Static type inference for tuun expressions.
//!
//! Implements the algorithmic type system of Xie and Oliveira, "Let Arguments
//! Go First" (ESOP 2018), §3.5 and Appendix E: bi-directional type checking
//! with an *application mode*. At an application, the argument types are
//! inferred first, generalized (rule AT-Gen), and pushed onto an application
//! context Ψ — a stack of pending applications; the applied function is then
//! typed under Ψ, so unannotated parameters take their types from the arguments
//! the function is actually applied to (rule AT-Lam2). Because the parser
//! desugars `let x = e in b` into `(fn(x) => b)(e)`, the generalization step
//! gives `let` bindings HM-style polymorphism without a dedicated rule (paper
//! §2.3 and §3.2, T-Let).
//!
//! The paper's algorithmic judgment is `(S, N) ∣ Γ ∣ Ψ ⊢ e ⇒ A ↪ (S', N')`
//! (Fig. 16), threading a substitution `S` of meta-variable solutions and a
//! fresh-name supply `N`; here both live in [`Infer`], mutated in place. Tuun
//! departs from the paper as follows:
//!
//! - Functions are n-ary and applied all at once, so each Ψ entry is a
//!   [`Frame`] holding a whole call's argument types (positional and named)
//!   rather than a single type.
//! - Judgments return only the *residual* result type, with Ψ's worth of
//!   parameters already consumed — the style of the paper's §4.2 (rule
//!   T-Lam-Alt), which Lemma 1 (Ψ coincides with typing results) justifies
//!   against the §3 presentation.
//! - Base coercions `float <: waveform` and `seq <: waveform` model the
//!   evaluator's promotion of floats to constant waveforms.
//! - The checker reports warnings, never errors: each failed check produces an
//!   [`Error`] and inference recovers with [`Type::Dynamic`] so one mistake
//!   does not cascade. Evaluation must not depend on the checker's verdict. The
//!   checker's report stands alone — unbound variables and unresolvable modules
//!   warn here even though evaluation reports them too — so that it can
//!   eventually gate evaluation. Parse errors are the one exception: the
//!   checker only runs on parsed programs.

use std::collections::HashMap;

use crate::expr::{Binding, Error, Expr, Pattern, SourceBinding, SourceExpr, Span};
use crate::signatures;
use crate::types::Type;

/// What a program's evaluated result is expected to be.
///
/// Mirrors the run-time kind check that evaluation performs on program
/// results, so the checker can warn about a mismatch before evaluation.
pub enum Expectation {
    /// The program must produce a waveform (a seq also qualifies).
    Waveform,
    /// The program must produce a note function: applied to a note number
    /// and a velocity (both floats), it must produce a pair of waveforms.
    NoteFunction,
}

/// Checks `expr` under `bindings`, resolving `open`/`use` through `resolve`,
/// and returns the warnings found.
///
/// The signature mirrors [`crate::eval::evaluate`]; `resolve` should behave
/// identically to the resolver evaluation uses. Warnings never block
/// anything: callers surface them and evaluate regardless.
///
/// A `NoteFunction` expectation is checked by seeding Ψ with a frame of two
/// floats — the application mode makes "will be called with a note number
/// and velocity" expressible as an ordinary pending application.
pub fn check_program<'a, M, S, F>(
    resolve: F,
    bindings: &'a [SourceBinding<M, S>],
    expr: &SourceExpr<M, S>,
    expectation: Option<Expectation>,
) -> Vec<Error<S>>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    S: Clone,
{
    let mut checker = Infer::new();
    let mut context = Vec::new();
    let mut memo = HashMap::new();
    checker.build_context(&resolve, bindings, &mut context, &mut memo);
    match expectation {
        None => {
            checker.infer(&mut context, &mut Vec::new(), expr);
        }
        Some(Expectation::Waveform) => {
            let ty = checker.infer(&mut context, &mut Vec::new(), expr);
            checker.subtype_check(&ty, &Type::Waveform, &expr.span);
        }
        Some(Expectation::NoteFunction) => {
            let mut psi = vec![Frame {
                positional: vec![
                    (Type::Float, expr.span.clone()),
                    (Type::Float, expr.span.clone()),
                ],
                named: Vec::new(),
                span: expr.span.clone(),
            }];
            let ty = checker.infer(&mut context, &mut psi, expr);
            let expected = Type::Tuple(vec![Type::Waveform, Type::Waveform]);
            checker.subtype_check(&ty, &expected, &expr.span);
        }
    }
    checker.warnings
}

/// One application's worth of argument types, awaiting the function they
/// apply to.
///
/// The paper's application context Ψ (§3.1) is a stack holding one argument
/// type per entry, because its applications are curried. Tuun applies all of
/// a call's arguments at once, so each Ψ entry here is a whole call:
/// positional argument types in order plus named arguments. Each argument
/// carries its own span so a mismatch can point at the offending argument;
/// `span` locates the call as a whole.
struct Frame<S> {
    positional: Vec<(Type, Option<Span<S>>)>,
    named: Vec<(String, Type, Option<Span<S>>)>,
    span: Option<Span<S>>,
}

/// The application context Ψ: pending applications, innermost call on top.
type Psi<S> = Vec<Frame<S>>;

/// A typing-context entry.
///
/// Built-in definitions stay name-resolved rather than being typed once at
/// their binding: their signature is looked up at each use, with the arity
/// of the call in Ψ, so arity-overloaded built-ins (unary versus binary `-`)
/// pick the form matching the call site. This keeps rule AT-Var's context
/// lookup happening where the paper performs it — at the variable, under Ψ.
#[derive(Clone)]
enum ContextEntry {
    Ty(Type),
    Builtin(String),
}

/// The typing context Γ: in-scope identifiers and their entries; later
/// entries shadow earlier ones, mirroring the evaluation context.
type TypeContext = Vec<(String, ContextEntry)>;

/// The algorithmic state threaded through every judgment: the paper's
/// substitution `S` and name supply `N` (Appendix E.1), plus the warnings
/// accumulated so far.
struct Infer<S> {
    /// Solutions for meta variables — the paper's `S`. Kept acyclic by the
    /// occurs check; solutions may mention other solved metas, so readers
    /// follow chains.
    subst: HashMap<u32, Type>,
    /// Fresh-id supply — the paper's `N`, shared by metas and rigid
    /// variables.
    supply: u32,
    warnings: Vec<Error<S>>,
}

impl<S: Clone> Infer<S> {
    fn new() -> Infer<S> {
        Infer {
            subst: HashMap::new(),
            supply: 0,
            warnings: Vec::new(),
        }
    }

    fn warn(&mut self, message: String, span: &Option<Span<S>>) {
        self.warnings.push(Error::with_span(message, span.clone()));
    }

    fn fresh_id(&mut self) -> u32 {
        self.supply += 1;
        self.supply
    }

    fn fresh_meta(&mut self) -> Type {
        Type::Meta(self.fresh_id())
    }

    /// Follows substitution chains at the root of `ty` only.
    fn resolve(&self, ty: &Type) -> Type {
        let mut ty = ty.clone();
        while let Type::Meta(id) = &ty {
            match self.subst.get(id) {
                Some(solution) => ty = solution.clone(),
                None => break,
            }
        }
        ty
    }

    /// Renders `ty` for a warning message, with solutions applied.
    fn display(&self, ty: &Type) -> String {
        ty.apply(&self.subst).to_string()
    }

    /// Replaces the quantified `vars` in `body` with fresh metas — the
    /// paper's `A[a ↦ β̂]` instantiation (rules AS-ForallL and AS-ForallL2,
    /// Fig. 15).
    fn instantiate(&mut self, vars: &[u32], body: Type) -> Type {
        let mut mapping = HashMap::new();
        for var in vars {
            mapping.insert(*var, self.fresh_meta());
        }
        body.substitute_vars(&mapping)
    }

    /// Generalizes `ty` under `context` — rule AT-Gen (Fig. 16): quantifies
    /// `ᾱ = ftv(S ty) − ftv(S context)`, replacing those metas with fresh rigid
    /// variables `b̄` to build `∀b̄.(S ty)[ᾱ ↦ b̄]`.
    ///
    /// Applied at every argument position (rule AT-App); with the parser's
    /// `let` desugaring this is exactly where HM let-generalization lands
    /// (§3.2, "Let Expressions"), and what allows arguments to have
    /// higher-rank types (§2.4).
    fn generalize(&mut self, context: &TypeContext, ty: Type) -> Type {
        let applied = ty.apply(&self.subst);
        let mut metas = Vec::new();
        applied.free_metas(&self.subst, &mut metas);
        if metas.is_empty() {
            return applied;
        }
        let mut context_metas = Vec::new();
        for (_, entry) in context.iter() {
            if let ContextEntry::Ty(ty) = entry {
                ty.free_metas(&self.subst, &mut context_metas);
            }
        }
        let quantify: Vec<u32> = metas
            .into_iter()
            .filter(|meta| !context_metas.contains(meta))
            .collect();
        if quantify.is_empty() {
            return applied;
        }
        let mut mapping = HashMap::new();
        let mut vars = Vec::new();
        for meta in &quantify {
            let var = self.fresh_id();
            vars.push(var);
            mapping.insert(*meta, Type::Var(var));
        }
        Type::Forall(vars, Box::new(applied.substitute_metas(&mapping)))
    }

    /// Unifies two types, solving metas by equality — Fig. 13. Fails
    /// silently; callers decide whether a failure warrants a warning.
    ///
    /// Structural cases generalize the paper's AU-Fun to tuun's n-ary
    /// functions, tuples, lists, and modules. `Dynamic` unifies with
    /// anything (a tuun extension; no counterpart in the paper).
    fn unify(&mut self, a: &Type, b: &Type) -> Result<(), ()> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        match (&a, &b) {
            (Type::Dynamic, _) | (_, Type::Dynamic) => Ok(()),
            // AU-Refl.
            (Type::Meta(x), Type::Meta(y)) if x == y => Ok(()),
            // Predicative (§2.4): a meta never holds a quantified type.
            (Type::Meta(_), Type::Forall(_, _)) | (Type::Forall(_, _), Type::Meta(_)) => Err(()),
            // AU-Var1/AU-Var2: solve an unsolved meta, under the occurs
            // check (`α̂ ∉ ftv(S τ)`). AU-BVar1/AU-BVar2 (already-solved
            // metas) are handled by `resolve` above.
            (Type::Meta(id), other) | (other, Type::Meta(id)) => {
                let mut metas = Vec::new();
                other.free_metas(&self.subst, &mut metas);
                if metas.contains(id) {
                    return Err(());
                }
                self.subst.insert(*id, other.clone());
                Ok(())
            }
            (Type::Var(x), Type::Var(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err(())
                }
            }
            // AU-Refl, extended to tuun's base types.
            (Type::Float, Type::Float)
            | (Type::Bool, Type::Bool)
            | (Type::String, Type::String)
            | (Type::Waveform, Type::Waveform)
            | (Type::Seq, Type::Seq) => Ok(()),
            // AU-Fun, n-ary: parameters (including named, matched by name)
            // and result unify pointwise.
            (
                Type::Function {
                    positional: p1,
                    named: n1,
                    result: r1,
                },
                Type::Function {
                    positional: p2,
                    named: n2,
                    result: r2,
                },
            ) => {
                if p1.len() != p2.len() || n1.len() != n2.len() {
                    return Err(());
                }
                for (x, y) in p1.iter().zip(p2) {
                    self.unify(x, y)?;
                }
                for (name, x) in n1 {
                    let Some((_, y)) = n2.iter().find(|(n, _)| n == name) else {
                        return Err(());
                    };
                    self.unify(x, y)?;
                }
                self.unify(r1, r2)
            }
            (Type::Tuple(xs), Type::Tuple(ys)) => {
                if xs.len() != ys.len() {
                    return Err(());
                }
                for (x, y) in xs.iter().zip(ys) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            (Type::List(x), Type::List(y)) => self.unify(x, y),
            (Type::Module(xs), Type::Module(ys)) => {
                if xs.len() != ys.len() {
                    return Err(());
                }
                for (name, x) in xs {
                    let Some((_, y)) = ys.iter().find(|(n, _)| n == name) else {
                        return Err(());
                    };
                    self.unify(x, y)?;
                }
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// Checks `a <: b` ("`a` is at least as polymorphic as `b`") — the
    /// subtyping judgment of Fig. 15, extended with tuun's base coercions.
    fn subtype(&mut self, a: &Type, b: &Type) -> Result<(), ()> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        match (&a, &b) {
            (Type::Dynamic, _) | (_, Type::Dynamic) => Ok(()),
            // AS-ForallR: freshen the quantified variables (skolemize),
            // check against the body, then require that no skolem leaked
            // into `a` (the rule's `b ∉ ftv(...)` side conditions) — a leak
            // would mean `a` is less polymorphic than promised.
            (_, Type::Forall(vars, body)) => {
                let mut mapping = HashMap::new();
                let mut skolems = Vec::new();
                for var in vars {
                    let id = self.fresh_id();
                    skolems.push(id);
                    mapping.insert(*var, Type::Var(id));
                }
                let body = body.substitute_vars(&mapping);
                self.subtype(&a, &body)?;
                for id in skolems {
                    if a.contains_var(id, &self.subst) {
                        return Err(());
                    }
                }
                Ok(())
            }
            // AS-ForallL: instantiate the left-hand quantifier with fresh
            // metas and continue.
            (Type::Forall(vars, body), _) => {
                let instance = self.instantiate(vars, (**body).clone());
                self.subtype(&instance, &b)
            }
            // Tuun's base coercions (not in the paper): the evaluator
            // promotes floats to constant waveforms, and a seq is usable as
            // the waveform it wraps.
            (Type::Float, Type::Waveform) | (Type::Seq, Type::Waveform) => Ok(()),
            // AS-FunR/AS-FunL collapse into one n-ary case: parameters are
            // contravariant, the result covariant. Every named parameter the
            // supertype promises must be offered by the subtype; extra named
            // parameters (all optional) are fine.
            (
                Type::Function {
                    positional: p1,
                    named: n1,
                    result: r1,
                },
                Type::Function {
                    positional: p2,
                    named: n2,
                    result: r2,
                },
            ) => {
                if p1.len() != p2.len() {
                    return Err(());
                }
                for (sub, sup) in p1.iter().zip(p2) {
                    self.subtype(sup, sub)?;
                }
                for (name, sup) in n2 {
                    let Some((_, sub)) = n1.iter().find(|(n, _)| n == name) else {
                        return Err(());
                    };
                    self.subtype(sup, sub)?;
                }
                self.subtype(r1, r2)
            }
            (Type::Tuple(xs), Type::Tuple(ys)) => {
                if xs.len() != ys.len() {
                    return Err(());
                }
                for (x, y) in xs.iter().zip(ys) {
                    self.subtype(x, y)?;
                }
                Ok(())
            }
            (Type::List(x), Type::List(y)) => self.subtype(x, y),
            // AS-Mono: everything else must unify.
            _ => self.unify(&a, &b),
        }
    }

    /// Checks `found <: expected` and reports a warning at `span` on
    /// failure, rolling back any partial solving from the failed attempt.
    fn subtype_check(&mut self, found: &Type, expected: &Type, span: &Option<Span<S>>) {
        let snapshot = self.subst.clone();
        if self.subtype(found, expected).is_err() {
            self.subst = snapshot;
            let message = format!(
                "expected {}, found {}",
                self.display(expected),
                self.display(found)
            );
            self.warn(message, span);
        }
    }

    /// Returns the least type both sides coerce to (a tuun extension; the paper
    /// has no join because it has no subtyping between base types): unification
    /// if the types agree, the waveform top of the coercion lattice for base
    /// mixes (mixed float/waveform/seq lists and branches are common),
    /// pointwise for lists and tuples, and `Dynamic` with a warning otherwise.
    /// Quantified types join at an instance.
    fn join(&mut self, a: Type, b: Type, span: &Option<Span<S>>) -> Type {
        let a = self.resolve(&a);
        let b = self.resolve(&b);
        if let Type::Forall(vars, body) = &a {
            let instance = self.instantiate(vars, (**body).clone());
            return self.join(instance, b, span);
        }
        if let Type::Forall(vars, body) = &b {
            let instance = self.instantiate(vars, (**body).clone());
            return self.join(a, instance, span);
        }
        match (&a, &b) {
            (Type::Dynamic, _) | (_, Type::Dynamic) => Type::Dynamic,
            (Type::Float, Type::Waveform)
            | (Type::Waveform, Type::Float)
            | (Type::Seq, Type::Waveform)
            | (Type::Waveform, Type::Seq)
            | (Type::Float, Type::Seq)
            | (Type::Seq, Type::Float) => Type::Waveform,
            (Type::List(x), Type::List(y)) => {
                let joined = self.join((**x).clone(), (**y).clone(), span);
                Type::List(Box::new(joined))
            }
            (Type::Tuple(xs), Type::Tuple(ys)) if xs.len() == ys.len() => {
                let items = xs
                    .iter()
                    .zip(ys)
                    .map(|(x, y)| self.join(x.clone(), y.clone(), span))
                    .collect();
                Type::Tuple(items)
            }
            _ => {
                let snapshot = self.subst.clone();
                if self.unify(&a, &b).is_ok() {
                    self.resolve(&a)
                } else {
                    self.subst = snapshot;
                    let message = format!(
                        "incompatible types {} and {}",
                        self.display(&a),
                        self.display(&b)
                    );
                    self.warn(message, span);
                    Type::Dynamic
                }
            }
        }
    }

    /// Applies `ty` to the pending applications in `psi`, returning the
    /// residual result type — the application subtyping judgment
    /// `(S, N) ∣ Ψ ⊢ A <: B` of Fig. 15 (bottom), with one frame per call.
    fn app_subtype(&mut self, psi: &mut Psi<S>, ty: Type) -> Type {
        // AS-Empty: not applied to anything, so the type stands as is.
        let Some(frame) = psi.pop() else { return ty };
        match self.resolve(&ty) {
            // AS-ForallL2: instantiate and keep consuming.
            Type::Forall(vars, body) => {
                let instance = self.instantiate(&vars, *body);
                psi.push(frame);
                self.app_subtype(psi, instance)
            }
            // AS-Fun2, n-ary: the frame's arguments feed one call; continue
            // with the rest of Ψ against the result type.
            Type::Function {
                positional,
                named,
                result,
            } => {
                self.check_frame(&frame, &positional, &named);
                self.app_subtype(psi, *result)
            }
            // AS-Mono2, via arrow unification (Fig. 14, AF-Mono): a meta
            // being applied must be a function; solve it to a function
            // shaped by the frame, then retry.
            meta @ Type::Meta(_) => {
                let positional = frame.positional.iter().map(|_| self.fresh_meta()).collect();
                let named = frame
                    .named
                    .iter()
                    .map(|(name, _, _)| (name.clone(), self.fresh_meta()))
                    .collect();
                let function = Type::Function {
                    positional,
                    named,
                    result: Box::new(self.fresh_meta()),
                };
                let _ = self.unify(&meta, &function);
                psi.push(frame);
                self.app_subtype(psi, function)
            }
            Type::Dynamic => {
                psi.clear();
                Type::Dynamic
            }
            // Applying a non-function: the static form of evaluation's
            // "Invalid application" error.
            other => {
                let message = format!("cannot apply a value of type {}", self.display(&other));
                self.warn(message, &frame.span);
                psi.clear();
                Type::Dynamic
            }
        }
    }

    /// Checks one call's arguments against a function type's parameters:
    /// arity, named-parameter existence, and argument subtyping (the
    /// premises of rule AS-Fun2, plus tuun's arity and named-argument checks,
    /// which evaluation performs at application time).
    fn check_frame(&mut self, frame: &Frame<S>, positional: &[Type], named: &[(String, Type)]) {
        if frame.positional.len() > positional.len() {
            self.warn("extra positional parameter".to_string(), &frame.span);
        } else if frame.positional.len() < positional.len() {
            let missing = self.display(&positional[frame.positional.len()]);
            self.warn(
                format!("missing parameter of type {}", missing),
                &frame.span,
            );
        }
        for ((argument, span), parameter) in frame.positional.iter().zip(positional) {
            self.subtype_check(argument, parameter, span);
        }
        for (name, argument, span) in &frame.named {
            match named.iter().find(|(n, _)| n == name) {
                Some((_, parameter)) => {
                    let parameter = parameter.clone();
                    self.subtype_check(argument, &parameter, span);
                }
                None => self.warn(format!("no named parameter \"{}\"", name), &frame.span),
            }
        }
    }

    /// Infers the type of `expr` under `context` — the typing judgment of
    /// Fig. 16 — consuming every pending application in `psi`; the returned
    /// type is the residual after all those applications (§4.2 style, per
    /// Lemma 1).
    fn infer<M>(
        &mut self,
        context: &mut TypeContext,
        psi: &mut Psi<S>,
        expr: &SourceExpr<M, S>,
    ) -> Type {
        match &expr.expr {
            // AT-Int, extended to tuun's literal forms. A literal under a
            // non-empty Ψ falls into `app_subtype`'s cannot-apply case.
            Expr::Bool(_) => self.app_subtype(psi, Type::Bool),
            Expr::Float(_) => self.app_subtype(psi, Type::Float),
            Expr::String(_) => self.app_subtype(psi, Type::String),
            Expr::Waveform(_) => self.app_subtype(psi, Type::Waveform),
            Expr::Seq { offset, waveform } => {
                let offset_ty = self.infer(context, &mut Vec::new(), offset);
                self.subtype_check(&offset_ty, &Type::Waveform, &offset.span);
                let waveform_ty = self.infer(context, &mut Vec::new(), waveform);
                self.subtype_check(&waveform_ty, &Type::Waveform, &waveform.span);
                self.app_subtype(psi, Type::Seq)
            }
            // AT-Var: look the variable up and apply its type to Ψ.
            Expr::Variable(name) => match context.iter().rev().find(|(n, _)| n == name) {
                Some((_, ContextEntry::Ty(ty))) => {
                    let ty = ty.clone();
                    self.app_subtype(psi, ty)
                }
                Some((_, ContextEntry::Builtin(builtin))) => {
                    let arity = psi.last().map(|frame| frame.positional.len());
                    let ty = signatures::signature(builtin, arity).unwrap_or(Type::Dynamic);
                    self.app_subtype(psi, ty)
                }
                None => {
                    self.warn(format!("unbound variable '{}'", name), &expr.span);
                    psi.clear();
                    Type::Dynamic
                }
            },
            // AT-Var for built-ins: the signature table stands in for the
            // typing-context entry.
            Expr::BuiltIn { name, .. } => {
                let arity = psi.last().map(|frame| frame.positional.len());
                let ty = signatures::signature(name, arity).unwrap_or(Type::Dynamic);
                self.app_subtype(psi, ty)
            }
            Expr::Function {
                positional,
                named,
                body,
            } => {
                // Named parameters take the types of their defaults, which
                // are inferred in the enclosing scope (mirroring their
                // once-at-definition evaluation). Not in the paper, which
                // has no named parameters.
                let named_types: Vec<(String, Type)> = named
                    .iter()
                    .map(|(name, default)| {
                        (name.clone(), self.infer(context, &mut Vec::new(), default))
                    })
                    .collect();
                let depth = context.len();
                let result = match psi.pop() {
                    // AT-Lam2 (the application mode's centerpiece):
                    // parameters take the argument types popped from Ψ, and
                    // the result is the body's type under the rest of Ψ.
                    Some(frame) => {
                        let arity_matches = frame.positional.len() == positional.len();
                        if frame.positional.len() > positional.len() {
                            self.warn("extra positional parameter".to_string(), &frame.span);
                        } else if frame.positional.len() < positional.len() {
                            let missing = &positional[frame.positional.len()];
                            self.warn(format!("missing parameter \"{}\"", missing), &frame.span);
                        }
                        for (index, pattern) in positional.iter().enumerate() {
                            let (ty, span) = if arity_matches {
                                frame.positional[index].clone()
                            } else {
                                // Don't let an arity mismatch cascade into
                                // per-argument mismatch warnings.
                                (Type::Dynamic, frame.span.clone())
                            };
                            self.bind_pattern(context, pattern, ty, &span, false);
                        }
                        for (name, argument, span) in &frame.named {
                            match named_types.iter().find(|(n, _)| n == name) {
                                Some((_, parameter)) => {
                                    let parameter = parameter.clone();
                                    self.subtype_check(argument, &parameter, span);
                                }
                                None => {
                                    self.warn(
                                        format!("no named parameter \"{}\"", name),
                                        &frame.span,
                                    );
                                }
                            }
                        }
                        for (name, ty) in &named_types {
                            context.push((name.clone(), ContextEntry::Ty(ty.clone())));
                        }
                        self.infer(context, psi, body)
                    }
                    // AT-Lam1: unapplied, so parameters get fresh metas,
                    // HM-style.
                    None => {
                        let param_types: Vec<Type> = positional
                            .iter()
                            .map(|pattern| self.pattern_param_type(context, pattern))
                            .collect();
                        for (name, ty) in &named_types {
                            context.push((name.clone(), ContextEntry::Ty(ty.clone())));
                        }
                        let body_ty = self.infer(context, &mut Vec::new(), body);
                        Type::Function {
                            positional: param_types,
                            named: named_types,
                            result: Box::new(body_ty),
                        }
                    }
                };
                context.truncate(depth);
                result
            }
            // AT-App: infer each argument in inference mode (empty Ψ),
            // generalize it (AT-Gen), push the call's frame, and type the
            // function under the extended Ψ. The recursive call's result is
            // already the application's type (§4.2 style).
            Expr::Application {
                function,
                positional,
                named,
            } => {
                let positional_types = positional
                    .iter()
                    .map(|argument| {
                        let ty = self.infer(context, &mut Vec::new(), argument);
                        (self.generalize(context, ty), argument.span.clone())
                    })
                    .collect();
                let named_types = named
                    .iter()
                    .map(|(name, argument)| {
                        let ty = self.infer(context, &mut Vec::new(), argument);
                        (
                            name.clone(),
                            self.generalize(context, ty),
                            argument.span.clone(),
                        )
                    })
                    .collect();
                psi.push(Frame {
                    positional: positional_types,
                    named: named_types,
                    span: expr.span.clone(),
                });
                self.infer(context, psi, function)
            }
            // Conditionals are not applied directly to Ψ: both branches are
            // inferred in inference mode, joined, and the joined type is
            // then applied (the subsumption direction of Lemma 2 — inference
            // mode results can always enter the application mode).
            Expr::IfThenElse {
                condition,
                then,
                else_,
            } => {
                let condition_ty = self.infer(context, &mut Vec::new(), condition);
                self.subtype_check(&condition_ty, &Type::Bool, &condition.span);
                let then_ty = self.infer(context, &mut Vec::new(), then);
                let else_ty = self.infer(context, &mut Vec::new(), else_);
                let joined = self.join(then_ty, else_ty, &expr.span);
                self.app_subtype(psi, joined)
            }
            // Pairs take the inference-mode rule (the paper's Pair-I, §2.1):
            // tuples cannot be applied, so components are inferred with an
            // empty Ψ.
            Expr::Tuple(items) => {
                let types = items
                    .iter()
                    .map(|item| self.infer(context, &mut Vec::new(), item))
                    .collect();
                self.app_subtype(psi, Type::Tuple(types))
            }
            Expr::List(items) => {
                let mut element: Option<Type> = None;
                for item in items {
                    let ty = self.infer(context, &mut Vec::new(), item);
                    element = Some(match element {
                        None => ty,
                        Some(previous) => self.join(previous, ty, &item.span),
                    });
                }
                let element = element.unwrap_or_else(|| self.fresh_meta());
                self.app_subtype(psi, Type::List(Box::new(element)))
            }
            Expr::Project { module, name } => {
                let module_ty = self.infer(context, &mut Vec::new(), module);
                let ty = match self.resolve(&module_ty) {
                    Type::Module(entries) => match entries.iter().find(|(n, _)| n == name) {
                        Some((_, ty)) => ty.clone(),
                        None => {
                            self.warn(format!("Module has no binding '{}'", name), &expr.span);
                            Type::Dynamic
                        }
                    },
                    // A module of unknown type (e.g. an unannotated
                    // parameter): projection could be fine at run time.
                    Type::Dynamic | Type::Meta(_) => Type::Dynamic,
                    other => {
                        let message = format!(
                            "cannot project '{}' from a value of type {}",
                            name,
                            self.display(&other)
                        );
                        self.warn(message, &expr.span);
                        Type::Dynamic
                    }
                };
                self.app_subtype(psi, ty)
            }
            Expr::BoundModule(entries) => {
                let types = entries
                    .iter()
                    .map(|(name, entry)| {
                        (name.clone(), self.infer(context, &mut Vec::new(), entry))
                    })
                    .collect();
                self.app_subtype(psi, Type::Module(types))
            }
            // The parser already reported the error.
            Expr::Error(_) => {
                psi.clear();
                Type::Dynamic
            }
        }
    }

    /// Binds the identifiers of `pattern` in `context` at the corresponding
    /// components of `ty`, destructuring tuple types as needed.
    ///
    /// The meta case is the tuple analog of arrow unification (Fig. 14,
    /// AF-Mono): an unknown being destructured must be a tuple of the
    /// pattern's width. With `generalize_leaves` (definitions), each bound
    /// identifier's type is generalized (AT-Gen) so tuple bindings stay
    /// polymorphic per component.
    fn bind_pattern(
        &mut self,
        context: &mut TypeContext,
        pattern: &Pattern,
        ty: Type,
        span: &Option<Span<S>>,
        generalize_leaves: bool,
    ) {
        match pattern {
            Pattern::Identifier(name) => {
                let ty = if generalize_leaves {
                    self.generalize(context, ty)
                } else {
                    ty
                };
                context.push((name.clone(), ContextEntry::Ty(ty)));
            }
            Pattern::Tuple(patterns) => match self.resolve(&ty) {
                Type::Tuple(items) if items.len() == patterns.len() => {
                    for (pattern, item) in patterns.iter().zip(items) {
                        self.bind_pattern(context, pattern, item, span, generalize_leaves);
                    }
                }
                meta @ Type::Meta(_) => {
                    let shape = Type::Tuple(patterns.iter().map(|_| self.fresh_meta()).collect());
                    let _ = self.unify(&meta, &shape);
                    self.bind_pattern(context, pattern, shape, span, generalize_leaves);
                }
                Type::Forall(vars, body) => {
                    let instance = self.instantiate(&vars, *body);
                    self.bind_pattern(context, pattern, instance, span, generalize_leaves);
                }
                Type::Dynamic => {
                    for pattern in patterns {
                        self.bind_pattern(context, pattern, Type::Dynamic, span, generalize_leaves);
                    }
                }
                other => {
                    let message = format!(
                        "pattern {} does not match type {}",
                        pattern,
                        self.display(&other)
                    );
                    self.warn(message, span);
                    for pattern in patterns {
                        self.bind_pattern(context, pattern, Type::Dynamic, span, generalize_leaves);
                    }
                }
            },
        }
    }

    /// Returns the parameter type for `pattern` in an unapplied function
    /// (rule AT-Lam1): a fresh meta per identifier, shaped by tuple
    /// patterns. Binds each identifier in `context`.
    fn pattern_param_type(&mut self, context: &mut TypeContext, pattern: &Pattern) -> Type {
        match pattern {
            Pattern::Identifier(name) => {
                let meta = self.fresh_meta();
                context.push((name.clone(), ContextEntry::Ty(meta.clone())));
                meta
            }
            Pattern::Tuple(patterns) => Type::Tuple(
                patterns
                    .iter()
                    .map(|pattern| self.pattern_param_type(context, pattern))
                    .collect(),
            ),
        }
    }

    /// Walks `bindings` accumulating types into `context`, mirroring evaluation's
    /// context building: `open` splices a module's own exports, `use` binds a
    /// module type, and definitions are inferred, generalized (AT-Gen, as at
    /// the top level of a `let`), and bound. Returns the walked bindings' own
    /// exports (excluding opened ones).
    ///
    /// An unresolvable module produces a warning at its binding; its names
    /// then warn as unbound wherever they are used.
    fn build_context<'a, M, F>(
        &mut self,
        resolve: &F,
        bindings: &'a [SourceBinding<M, S>],
        context: &mut TypeContext,
        memo: &mut HashMap<usize, Vec<(String, ContextEntry)>>,
    ) -> Vec<(String, ContextEntry)>
    where
        F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    {
        let mut own = Vec::new();
        for source_binding in bindings {
            match &source_binding.binding {
                Binding::Open(path) => match resolve(path) {
                    Ok(module) => {
                        let exports = self.module_exports(resolve, module, memo);
                        context.extend(exports);
                    }
                    Err(error) => {
                        let message = format!(
                            "cannot load module '{}': {}",
                            path.join("."),
                            error.message()
                        );
                        self.warn(message, &source_binding.span);
                    }
                },
                Binding::Use(path) => {
                    let Some(name) = path.last() else { continue };
                    let module = match resolve(path) {
                        Ok(module) => module,
                        Err(error) => {
                            let message = format!(
                                "cannot load module '{}': {}",
                                path.join("."),
                                error.message()
                            );
                            self.warn(message, &source_binding.span);
                            continue;
                        }
                    };
                    let exports = self.module_exports(resolve, module, memo);
                    // The module type holds plain types; a re-exported
                    // built-in is fixed at its default signature.
                    let members = dedup_last_wins(exports)
                        .into_iter()
                        .map(|(member, entry)| {
                            let ty = match entry {
                                ContextEntry::Ty(ty) => ty,
                                ContextEntry::Builtin(builtin) => {
                                    signatures::signature(&builtin, None).unwrap_or(Type::Dynamic)
                                }
                            };
                            (member, ty)
                        })
                        .collect();
                    context.push((name.clone(), ContextEntry::Ty(Type::Module(members))));
                }
                Binding::Definition(pattern, expr) => {
                    let before = context.len();
                    // A built-in bound to a name stays name-resolved (see
                    // `ContextEntry`) so its signature can adapt to each call's
                    // arity.
                    if let (Pattern::Identifier(name), Expr::BuiltIn { name: builtin, .. }) =
                        (pattern, &expr.expr)
                    {
                        context.push((name.clone(), ContextEntry::Builtin(builtin.clone())));
                    } else {
                        let ty = self.infer(context, &mut Vec::new(), expr);
                        self.bind_pattern(context, pattern, ty, &expr.span, true);
                    }
                    own.extend_from_slice(&context[before..]);
                }
                Binding::Empty => {}
            }
        }
        own
    }

    /// Returns the exports of `module`, typing it in a fresh context.
    ///
    /// Memoized per bindings slice so a module opened many times (the
    /// prelude, in particular) is typed once per check. Sound because
    /// exports are generalized and thus closed under the substitution.
    fn module_exports<'a, M, F>(
        &mut self,
        resolve: &F,
        module: &'a [SourceBinding<M, S>],
        memo: &mut HashMap<usize, Vec<(String, ContextEntry)>>,
    ) -> Vec<(String, ContextEntry)>
    where
        F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    {
        let key = module.as_ptr() as usize;
        if let Some(exports) = memo.get(&key) {
            return exports.clone();
        }
        let mut module_context = Vec::new();
        let exports = self.build_context(resolve, module, &mut module_context, memo);
        memo.insert(key, exports.clone());
        exports
    }
}

/// Returns `entries` with duplicate names collapsed so each name appears
/// once, bound to the value of its last occurrence.
fn dedup_last_wins<T>(entries: Vec<(String, T)>) -> Vec<(String, T)> {
    let mut out: Vec<(String, T)> = Vec::new();
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
    use std::rc::Rc;

    use crate::builtins;
    use crate::expr::BuiltInFn;
    use crate::modules;
    use crate::parser::{parse_module, parse_program};

    /// Builds a prelude mirroring the native one: the built-ins plus
    /// `tempo`, `sample_rate`, `mark`, and `debug`. The extra built-ins get
    /// stub closures — the checker only looks at their names.
    fn test_prelude<S: Clone + std::fmt::Debug + 'static>() -> Vec<SourceBinding<u32, S>> {
        let mut prelude: Vec<SourceBinding<u32, S>> = Vec::new();
        builtins::add_bindings(&mut prelude);
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("tempo".to_string()),
            SourceExpr::float(120.0),
        ));
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("sample_rate".to_string()),
            SourceExpr::float(44100.0),
        ));
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("mark".to_string()),
            SourceExpr::from(Expr::BuiltIn {
                name: "mark".to_string(),
                function: BuiltInFn(Rc::new(|_| Expr::Error("stub".to_string()))),
            }),
        ));
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("debug".to_string()),
            builtins::debug(|_| {}),
        ));
        prelude
    }

    /// Parses and checks `input` with the prelude in scope and no
    /// expectation on the program's type.
    fn check(input: &str) -> Vec<Error<()>> {
        check_with_expectation(input, None)
    }

    fn check_with_expectation(input: &str, expectation: Option<Expectation>) -> Vec<Error<()>> {
        let bindings = test_prelude();
        let expr = parse_program::<u32, _>(input, ()).unwrap();
        check_program(
            |_: &[String]| Err(Error::new("no modules".to_string())),
            &bindings,
            &expr,
            expectation,
        )
    }

    fn messages(warnings: &[Error<()>]) -> Vec<String> {
        warnings.iter().map(|w| w.message().to_string()).collect()
    }

    #[track_caller]
    fn assert_clean(input: &str) {
        let warnings = check(input);
        assert!(
            warnings.is_empty(),
            "expected no warnings for {:?}, got {:?}",
            input,
            messages(&warnings)
        );
    }

    #[track_caller]
    fn assert_warns(input: &str, expected: &[&str]) {
        let warnings = check(input);
        assert_eq!(
            messages(&warnings),
            expected.to_vec(),
            "for input {:?}",
            input
        );
    }

    // The paper's motivating example (§2.2): an unannotated lambda,
    // typeable only because the argument type flows in via Ψ.
    #[test]
    fn applied_unannotated_lambda() {
        assert_clean("(fn(x) => x)(1)");
    }

    // HM-style let polymorphism through the parser's desugaring of `let`
    // into an applied lambda (§2.3, §3.2): `id` is generalized at the
    // application and used at two different types.
    #[test]
    fn let_polymorphism_through_desugaring() {
        assert_clean("let id = fn(x) => x in (id(1), id(\"s\"))");
    }

    // Higher-rank flow (§2.4): the lambda argument is generalized before
    // being pushed, so `g` is bound at a polymorphic type with no
    // annotation anywhere. Not typeable in HM or plain bi-directional
    // checking.
    #[test]
    fn lambda_bound_polymorphic_argument() {
        assert_clean("(fn(g) => (g(1), g(\"s\")))(fn(x) => x)");
    }

    // Contrast: a lambda inferred in inference mode (as `map`'s argument,
    // rule AT-Lam1) gets a monotype parameter, so using it at two types
    // warns.
    #[test]
    fn inference_mode_lambda_is_monomorphic() {
        assert_warns(
            "map(fn(g) => (g(1), g(\"s\")), [fn(x) => x])",
            &["expected float, found string"],
        );
    }

    // List covariance plus the float <: waveform coercion, at the end of a
    // chain that starts inside the lambda: `+`'s single signature solves
    // `x`'s meta to waveform, so the lambda is (waveform) -> waveform;
    // checking it against map's ∀a instantiation (frame arguments go left
    // to right) pins `a` at waveform; only then is `[1, 2] : [float]`
    // compared against `[a]` = [waveform], which needs both List
    // covariance and the element coercion to pass.
    #[test]
    fn map_with_coercions() {
        assert_clean("map(fn(x) => x + 1, [1, 2])");
    }

    #[test]
    fn reduce_polymorphism() {
        assert_clean("reduce(fn(acc, x) => acc + x, 0, [1, 2, 3])");
    }

    #[test]
    fn builtin_passed_as_argument() {
        // `sqrt` is referenced under an empty Ψ (rule AS-Empty): its
        // signature flows to `map` unapplied.
        assert_clean("map(sqrt, [1, 2])");
    }

    // A mismatched argument warns and the warning's span points at the
    // offending argument, not the whole call.
    #[test]
    fn argument_mismatch_points_at_argument() {
        let input = "sine(\"a\", 0)";
        let warnings = check(input);
        assert_eq!(messages(&warnings), ["expected waveform, found string"]);
        let start = input.find("\"a\"").unwrap();
        assert_eq!(warnings[0].range(), Some(start..start + 3));
    }

    #[test]
    fn tuple_patterns() {
        assert_clean("let (a, b) = (1, \"x\") in a");
        // Two separate arguments do not destructure into a tuple pattern;
        // the arity warning alone reports it (no per-argument cascade).
        assert_warns("(fn((y, z)) => y)(4, 5)", &["extra positional parameter"]);
    }

    // Named parameters take their types from their defaults; call-site
    // checks mirror evaluation's (unknown name, extra positional).
    #[test]
    fn named_parameters() {
        let f = "let f = fn(x, y = 10) => x * y in ";
        assert_clean(&format!("{}f(2)", f));
        assert_clean(&format!("{}f(2, y = 5)", f));
        assert_warns(&format!("{}f(2, z = 3)", f), &["no named parameter \"z\""]);
        assert_warns(&format!("{}f(2, 3)", f), &["extra positional parameter"]);
        // A named argument to a builtin without named parameters warns for
        // the bogus name, and — since only one of sine's two positional
        // parameters is supplied — for the missing argument too.
        assert_warns(
            "sine(440, y = 1)",
            &[
                "missing parameter of type waveform",
                "no named parameter \"y\"",
            ],
        );
    }

    #[test]
    fn missing_argument() {
        assert_warns("(fn(x, y) => x)(2)", &["missing parameter \"y\""]);
    }

    #[test]
    fn conditionals() {
        assert_warns("if 1 then 2 else 3", &["expected bool, found float"]);
        // Branches join over the coercion lattice: float and waveform meet
        // at waveform.
        assert_clean("if true then 1 else sine(440, 0)");
    }

    // KNOWN SPURIOUS WARNING, pinned deliberately: `+` has the single
    // signature (waveform, waveform) -> waveform, so `1 + 2` types as
    // waveform even though it evaluates to a float, and `nth` then rejects
    // it. Iterating the signature design (e.g. refinement types) should
    // eventually turn this test green-with-no-warnings.
    #[test]
    fn known_spurious_arithmetic_widens_to_waveform() {
        assert_warns("nth(1 + 2, [1, 2, 3])", &["expected float, found waveform"]);
    }

    #[test]
    fn seq_typing() {
        // seq(0)(w) builds a seq; `\` wants a seq on the left.
        assert_clean("seq(0)(sine(440, 0)) \\ 1");
        // A bare waveform on the left of `\` is a genuine runtime error.
        assert_warns("sine(440, 0) \\ 1", &["expected seq, found waveform"]);
        // KNOWN SPURIOUS WARNING, pinned deliberately: arithmetic on a seq
        // types as waveform (the single binary-op signature erases
        // seq-ness), so the following `\` warns even though evaluation
        // keeps the seq.
        assert_warns(
            "(seq(0)(sine(440, 0)) * 0.5) \\ 1",
            &["expected seq, found waveform"],
        );
    }

    #[test]
    fn unbound_variable_warns() {
        let input = "nope + 1";
        let warnings = check(input);
        assert_eq!(messages(&warnings), ["unbound variable 'nope'"]);
        let start = input.find("nope").unwrap();
        assert_eq!(warnings[0].range(), Some(start..start + 4));

        assert_warns("nope(1) * 2", &["unbound variable 'nope'"]);
        // An unbound name recovers as Dynamic, so it produces exactly one
        // warning even when applied and used in arithmetic.
        assert_warns(
            "sine(missing, wrong)",
            &["unbound variable 'missing'", "unbound variable 'wrong'"],
        );
    }

    // Unresolvable open/use bindings warn at the binding; the module's
    // names then warn as unbound at their use sites.
    #[test]
    fn unresolvable_modules_warn() {
        let mut bindings = test_prelude();
        let (opens, errors) = parse_module::<u32, _>("open nope;\nuse also.missing;", ()).unwrap();
        assert!(errors.is_empty());
        bindings.extend(opens);
        let expr = parse_program::<u32, _>("something_from_nope + 1", ()).unwrap();
        let warnings = check_program(
            |_: &[String]| Err(Error::new("no modules".to_string())),
            &bindings,
            &expr,
            None,
        );
        assert_eq!(
            messages(&warnings),
            [
                "cannot load module 'nope': no modules",
                "cannot load module 'also.missing': no modules",
                "unbound variable 'something_from_nope'",
            ]
        );
    }

    #[test]
    fn program_expectations() {
        let warnings = check_with_expectation("\"hello\"", Some(Expectation::Waveform));
        assert_eq!(messages(&warnings), ["expected waveform, found string"]);
        let warnings = check_with_expectation("sine(440, 0)", Some(Expectation::Waveform));
        assert!(warnings.is_empty(), "got {:?}", messages(&warnings));
        // A seq qualifies as a waveform program.
        let warnings = check_with_expectation("seq(0)(sine(440, 0))", Some(Expectation::Waveform));
        assert!(warnings.is_empty(), "got {:?}", messages(&warnings));
    }

    // A keys program is checked by seeding Ψ with (float, float) — the
    // note number and velocity the runtime will invoke it with — and
    // requiring a pair of waveforms back.
    #[test]
    fn note_function_expectation() {
        let warnings = check_with_expectation(
            "fn(note, vel) => (sine(note, 0), fin(1)(sine(440, 0)))",
            Some(Expectation::NoteFunction),
        );
        assert!(warnings.is_empty(), "got {:?}", messages(&warnings));

        let warnings = check_with_expectation(
            "fn(note) => (sine(note, 0), sine(note, 0))",
            Some(Expectation::NoteFunction),
        );
        assert_eq!(messages(&warnings), ["extra positional parameter"]);

        let warnings = check_with_expectation(
            "fn(note, vel) => (sine(note, 0), \"x\")",
            Some(Expectation::NoteFunction),
        );
        assert_eq!(
            messages(&warnings),
            ["expected (waveform, waveform), found (waveform, string)"]
        );
    }

    // Mirrors eval's test_opens_are_scoped: opened names are usable inside
    // the opening module but not re-exported through it.
    #[test]
    fn open_is_scoped() {
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
        let mut bindings = test_prelude();
        let (opens, errors) = parse_module::<u32, _>("open a;", ()).unwrap();
        assert!(errors.is_empty());
        bindings.extend(opens);

        // `alias` is exported by `a` (bound there through `a`'s own open of
        // `b`); using it as a float checks.
        let expr = parse_program::<u32, _>("alias * 2", ()).unwrap();
        let warnings = check_program(resolve, &bindings, &expr, None);
        assert!(warnings.is_empty(), "got {:?}", messages(&warnings));

        // `two` is not re-exported through `a`, so it warns as unbound.
        let expr = parse_program::<u32, _>("two", ()).unwrap();
        let warnings = check_program(resolve, &bindings, &expr, None);
        assert_eq!(messages(&warnings), ["unbound variable 'two'"]);
    }

    // Mirrors eval's use/projection tests: `use` binds a module type,
    // projections type-check against it, and a missing member warns.
    #[test]
    fn use_and_projection() {
        let (b, errors) = parse_module::<u32, _>("two = 2;", ()).unwrap();
        assert!(errors.is_empty());
        let resolve = |path: &[String]| {
            if path == ["b"] {
                Ok(&b[..])
            } else {
                Err(Error::new(format!("no module {:?}", path)))
            }
        };
        let mut bindings = test_prelude();
        let (uses, errors) = parse_module::<u32, _>("use b;", ()).unwrap();
        assert!(errors.is_empty());
        bindings.extend(uses);

        let expr = parse_program::<u32, _>("b.two + 1", ()).unwrap();
        let warnings = check_program(resolve, &bindings, &expr, None);
        assert!(warnings.is_empty(), "got {:?}", messages(&warnings));

        let expr = parse_program::<u32, _>("b.three", ()).unwrap();
        let warnings = check_program(resolve, &bindings, &expr, None);
        assert_eq!(messages(&warnings), ["Module has no binding 'three'"]);

        // Projecting from a non-module warns statically too.
        let expr = parse_program::<u32, _>("let x = 1 in x.y", ()).unwrap();
        let warnings = check_program(resolve, &bindings, &expr, None);
        assert_eq!(
            messages(&warnings),
            ["cannot project 'y' from a value of type float"]
        );
    }

    #[test]
    fn cannot_apply_non_function() {
        assert_warns("1(2)", &["cannot apply a value of type float"]);
    }

    // Direct unit tests for the subtyping corners that full programs
    // rarely reach.
    #[test]
    fn subtype_corners() {
        let mut checker: Infer<()> = Infer::new();

        // ∀a.(a) -> a <: (float) -> float: instantiate left (AS-ForallL).
        let id = Type::Forall(
            vec![0],
            Box::new(Type::function(vec![Type::Var(0)], Type::Var(0))),
        );
        let mono = Type::function(vec![Type::Float], Type::Float);
        assert!(checker.subtype(&id, &mono).is_ok());

        // (float) -> float <: ∀a.(a) -> a: skolemize right (AS-ForallR)
        // and fail — the monotype is not that polymorphic.
        assert!(checker.subtype(&mono, &id).is_err());

        // A meta cannot absorb a skolem (the escape check): ?m <: ∀a.a.
        let meta = checker.fresh_meta();
        let all = Type::Forall(vec![0], Box::new(Type::Var(0)));
        assert!(checker.subtype(&meta, &all).is_err());

        // Occurs check (AU-Var1's side condition): ?m = [?m] must fail.
        let mut checker: Infer<()> = Infer::new();
        let meta = checker.fresh_meta();
        let list = Type::List(Box::new(meta.clone()));
        assert!(checker.unify(&meta, &list).is_err());
    }

    // Pins the checker's verdict on every embedded library module, so signature
    // or inference changes that affect the library surface here.
    //
    // Each module is parsed with its index as span source; a module's report
    // keeps only warnings from its own text (dependencies re-typed along the
    // way report under their own entry). Positions come from
    // `display_with_source` so every pinned warning can be read against the
    // .tuun source.
    #[test]
    fn embedded_modules_pinned_warnings() {
        let prelude = test_prelude::<u32>();
        let mut parsed: Vec<(String, &str, Vec<SourceBinding<u32, u32>>)> = Vec::new();
        for (index, (path, content)) in modules::EMBEDDED_MODULES.iter().enumerate() {
            let (mut bindings, errors) = parse_module::<u32, _>(content, index as u32).unwrap();
            assert!(errors.is_empty(), "parse errors in {}", path);
            bindings.insert(0, Binding::Open(vec!["__prelude".to_string()]).into());
            parsed.push((path.to_string(), content, bindings));
        }
        let resolve = |path: &[String]| {
            let key = path.join(".");
            if key == "__prelude" {
                return Ok(&prelude[..]);
            }
            parsed
                .iter()
                .find(|(name, _, _)| *name == key)
                .map(|(_, _, bindings)| &bindings[..])
                .ok_or_else(|| Error::new(format!("no module {}", key)))
        };
        let expr = parse_program::<u32, _>("0", 9999).unwrap();
        let mut report: Vec<String> = Vec::new();
        for (index, (path, content, bindings)) in parsed.iter().enumerate() {
            let warnings = check_program(&resolve, bindings, &expr, None);
            for warning in &warnings {
                if warning.source() == Some(index as u32) {
                    report.push(format!(
                        "{}: {}",
                        path,
                        warning.display_with_source(content)
                    ));
                }
            }
        }
        // Every pinned warning is a KNOWN SPURIOUS one of two documented
        // classes, verified against the library source:
        // - Arithmetic widening: the single binary-op signatures type any
        //   arithmetic as waveform, so float-only consumers (`exp`,
        //   `unfold`'s count) warn on computed floats like `-x` or `n+1`.
        // - Equality solving: metas solve by unification, not lattice join,
        //   so `ws` pinned at [seq] rejects a later [waveform] use.
        // Signature/inference improvements should shrink this list; anything
        // NEW appearing here needs the same source-level verification.
        assert_eq!(
            report,
            [
                "std: 13:31: expected float, found waveform",
                "std: 55:41: expected float, found waveform",
                "std: 58:19: expected [float], found [waveform]",
                "std: 143:18: expected [seq], found [waveform]",
                "filters.basic: 5:44: expected float, found waveform",
            ]
        );
    }
}
