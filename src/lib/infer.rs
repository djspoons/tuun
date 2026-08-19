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
//! - Numerics are refined by [`crate::types::Sort`]s — unions of four
//!   disjoint atoms — judged by containment, with overloaded functions
//!   typed as intersections of arrows selected per call (Freeman &
//!   Pfenning, "Refinement Types for ML", PLDI 1991; Xue, Oliveira & Xie,
//!   "Applicative Intersection Types"). A float *is* a constant waveform
//!   (`Waveform::Const` — see [`Expr::as_const_float`]), so `Float ⊆
//!   waveform` is genuine subset inclusion.
//! - Each failed check produces an [`Error`] and inference recovers with
//!   [`Type::Dynamic`] so one mistake does not cascade.

use std::collections::HashMap;

use crate::expr::{Binding, Error, Expr, Pattern, SourceBinding, SourceExpr, Span};
use crate::signatures;
use crate::types::{Refinement, Sort, Type};
use crate::waveform;

/// What a program's evaluated result is expected to be.
///
/// Mirrors the run-time kind check that evaluation performs on program
/// results, so the checker can reject a mismatch before evaluation.
pub enum Expectation {
    /// The program must produce a waveform or a seq.
    Playable,
    /// The program must produce a note function: applied to a note number
    /// (int) and a velocity (float), it must produce a pair of waveforms.
    NoteFunction,
}

/// Checks `expr` under `bindings`, resolving `open`/`use` through `resolve`,
/// and returns the errors found.
///
/// The signature mirrors [`crate::eval::evaluate`]; `resolve` should behave
/// identically to the resolver evaluation uses.
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
        Some(Expectation::Playable) => {
            let ty = checker.infer(&mut context, &mut Vec::new(), expr);
            checker.subtype_check(&ty, &Type::numeric(), &expr.span);
        }
        Some(Expectation::NoteFunction) => {
            let mut psi = vec![Frame {
                positional: vec![
                    (Type::int(), expr.span.clone()),
                    (Type::float(), expr.span.clone()),
                ],
                named: Vec::new(),
                span: expr.span.clone(),
            }];
            let ty = checker.infer(&mut context, &mut psi, expr);
            // The runtime requires a pair whose elements are waveform values.
            let expected = Type::Tuple(vec![Type::waveform(), Type::waveform()]);
            checker.subtype_check(&ty, &expected, &expr.span);
        }
    }
    checker.errors
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

/// The most atom vectors `tabulate` will enumerate for one definition:
/// 4^4, covering functions of up to four numeric parameters.
const MAX_TABULATION_VECTORS: usize = 256;

/// How selection decided one intersection against one frame.
enum Selection {
    /// A row applied; the application's result type.
    Selected(Type),
    /// No conjunct has the frame's shape; selection does not apply, and callers
    /// fall back to plain conjunct subtyping.
    NoMatchingRows,
    /// Conjuncts match the frame's shape but none accepts the arguments —
    /// the runtime has no matching arm.
    NoApplicableRow,
}

/// The algorithmic state threaded through every judgment: Xie and
/// Oliveira's substitution `S` and name supply `N` (Appendix E.1), plus the
/// errors accumulated so far.
struct Infer<S> {
    /// Solutions for meta variables — the paper's `S`. Kept acyclic by the
    /// occurs check; solutions may mention other solved metas, so readers
    /// follow chains.
    subst: HashMap<u32, Type>,
    /// Fresh-id supply — the paper's `N`, shared by metas and rigid
    /// variables.
    supply: u32,
    /// Bounds for refinement variables, indexed by `Refinement::Var` id.
    refs: Vec<RefVar>,
    /// The undo journal: every solver step since the start of the check,
    /// so failed attempts roll back by popping (see `mark`/`rollback`).
    journal: Vec<Undo>,
    errors: Vec<Error<S>>,
}

/// The bounds of one refinement variable: the join of the guarantees that
/// flowed into it (lower) and the meet of the contracts imposed on it (upper).
/// `may` reads the lower when any guarantee arrived, else the upper, else ⊤ —
/// the atoms the value may turn out to inhabit.
///
/// The discipline is the constraint range `S <: X <: T` of Pierce and
/// Turner's "Local Type Inference" (POPL 1998) — combining constraints
/// joins lower bounds and meets upper bounds — over the finite sort
/// lattice; MLsub's polar bounds (Dolan & Mycroft, POPL 2017) are the
/// full-strength form.
#[derive(Clone, Copy)]
struct RefVar {
    lower: Sort,
    upper: Option<Sort>,
}

/// One reversible solver step, recorded in the journal so a failed attempt can
/// be undone without cloning the whole state.
enum Undo {
    /// A meta was solved; undoing removes the solution.
    Solved(u32),
    /// A refinement variable was allocated; undoing pops it.
    Allocated,
    /// A refinement variable's bounds changed; undoing restores them.
    Bounds(u32, RefVar),
}

/// A point in the solver's history; everything after it can be rolled back.
/// Marks nest LIFO: roll back an inner mark before an outer one.
struct Mark(usize);

impl<S: Clone> Infer<S> {
    fn new() -> Infer<S> {
        Infer {
            subst: HashMap::new(),
            supply: 0,
            refs: Vec::new(),
            journal: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn error(&mut self, message: String, span: &Option<Span<S>>) {
        self.errors.push(Error::with_span(message, span.clone()));
    }

    fn fresh_id(&mut self) -> u32 {
        self.supply += 1;
        self.supply
    }

    fn fresh_meta(&mut self) -> Type {
        Type::Meta(self.fresh_id())
    }

    /// The current point in the solver's history, for `rollback`.
    fn mark(&self) -> Mark {
        Mark(self.journal.len())
    }

    /// Undoes every solver step recorded since `mark`.
    fn rollback(&mut self, mark: Mark) {
        while self.journal.len() > mark.0 {
            match self.journal.pop().expect("journal is non-empty") {
                Undo::Solved(id) => {
                    self.subst.remove(&id);
                }
                Undo::Allocated => {
                    self.refs.pop();
                }
                Undo::Bounds(id, var) => self.refs[id as usize] = var,
            }
        }
    }

    /// Solves a meta, recording the step for rollback.
    fn solve(&mut self, id: u32, ty: Type) {
        let previous = self.subst.insert(id, ty);
        debug_assert!(previous.is_none(), "metas are solved only once");
        self.journal.push(Undo::Solved(id));
    }

    /// Allocates a fresh, unconstrained refinement variable.
    fn fresh_refinement(&mut self) -> Refinement {
        self.refs.push(RefVar {
            lower: Sort::NONE,
            upper: None,
        });
        self.journal.push(Undo::Allocated);
        Refinement::Var((self.refs.len() - 1) as u32)
    }

    /// The sort a numeric may turn out to inhabit.
    fn may_of(&self, rep: &Refinement) -> Sort {
        match rep {
            Refinement::Ground(sort) => *sort,
            Refinement::Var(id) => {
                let var = &self.refs[*id as usize];
                if !var.lower.is_empty() {
                    var.lower
                } else {
                    var.upper.unwrap_or(Sort::TOP)
                }
            }
        }
    }

    /// The sort a numeric position requires (its contract).
    fn contract_of(&self, rep: &Refinement) -> Sort {
        match rep {
            Refinement::Ground(sort) => *sort,
            Refinement::Var(id) => self.refs[*id as usize].upper.unwrap_or(Sort::TOP),
        }
    }

    /// Records a guarantee flowing into a refinement variable; fails when the
    /// grown guarantees escape the contracts already imposed. Bounds only
    /// tighten, so the conflict is final — checking at both mutation sites
    /// keeps every variable's bounds consistent continuously, with no deferred
    /// re-validation pass.
    fn join_lower(&mut self, id: u32, sort: Sort) -> Result<(), ()> {
        let var = self.refs[id as usize];
        let joined = var.lower.union(sort);
        if joined == var.lower {
            return Ok(());
        }
        if let Some(upper) = var.upper
            && !joined.is_subset(upper)
        {
            return Err(());
        }
        self.journal.push(Undo::Bounds(id, var));
        self.refs[id as usize].lower = joined;
        Ok(())
    }

    /// Records a contract imposed on a refinement variable; fails when the
    /// combined contracts admit no value at all.
    fn meet_upper(&mut self, id: u32, sort: Sort) -> Result<(), ()> {
        let var = self.refs[id as usize];
        let met = var.upper.map_or(sort, |upper| upper.intersect(sort));
        if met.is_empty() {
            return Err(());
        }
        // Guarantees only grow and contracts only shrink, so a lower bound
        // escaping the met contract is a final violation.
        if !var.lower.is_subset(met) {
            return Err(());
        }
        if var.upper != Some(met) {
            self.journal.push(Undo::Bounds(id, var));
            self.refs[id as usize].upper = Some(met);
        }
        Ok(())
    }

    /// Checks that a found numeric satisfies an expected numeric — a ground
    /// sort by containment, may(found) ⊆ contract(expected); an unsolved
    /// variable by its guarantees so far, with the full judgment deferred to
    /// its recorded contracts — then records the flows: the found side's
    /// definite sort joins the expected side's guarantees, and the expected
    /// side's contract bounds the found side.
    fn numeric_subtype(&mut self, found: &Refinement, expected: &Refinement) -> Result<(), ()> {
        let may = self.may_of(found);
        let contract = self.contract_of(expected);
        match found {
            // A ground sort is judged by containment.
            Refinement::Ground(_) => {
                if !may.is_subset(contract) {
                    return Err(());
                }
            }
            // An unsolved variable is recorded rather than judged — the
            // guarantees seen so far under-approximate the final ones, so only
            // a violation by the current lower bound is final (lower bounds
            // only grow). The full judgment is deferred to the recorded
            // contracts.
            Refinement::Var(id) => {
                if !self.refs[*id as usize].lower.is_subset(contract) {
                    return Err(());
                }
            }
        }
        if let Refinement::Var(id) = expected {
            let definite = self.definite_of(found);
            self.join_lower(*id, definite)?;
        }
        if let Refinement::Var(id) = found {
            self.meet_upper(*id, contract)?;
        }
        Ok(())
    }

    /// The atoms a numeric definitely may inhabit: its sort when ground,
    /// its guarantees so far when unsolved. Unlike `may_of` there is no ⊤
    /// fallback — lower bounds hold only definite atoms, so they can be
    /// judged as final guarantees.
    fn definite_of(&self, rep: &Refinement) -> Sort {
        match rep {
            Refinement::Ground(sort) => *sort,
            Refinement::Var(id) => self.refs[*id as usize].lower,
        }
    }

    /// Merges two numerics met in an invariant position: each side's
    /// definite guarantees flow into the other, failing when they conflict
    /// with recorded contracts.
    // TODO two unsolved variables are not linked: guarantees or contracts
    // arriving at one after this merge do not reach the other. A union-find
    // over refinement variables would close this — MLsub's biunification
    // handles a variable-variable constraint exactly so (merge the bounds,
    // alias the variables).
    fn numeric_unify(&mut self, x: &Refinement, y: &Refinement) -> Result<(), ()> {
        let definite_x = self.definite_of(x);
        let definite_y = self.definite_of(y);
        if let Refinement::Var(id) = y {
            self.join_lower(*id, definite_x)?;
        }
        if let Refinement::Var(id) = x {
            self.join_lower(*id, definite_y)?;
        }
        Ok(())
    }

    /// Returns `ty` with refinement variables replaced by the sorts they may
    /// currently inhabit, for rendering.
    fn resolve_refinements(&self, ty: &Type) -> Type {
        match ty {
            Type::Numeric(rep @ Refinement::Var(_)) => Type::ground(self.may_of(rep)),
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional
                    .iter()
                    .map(|t| self.resolve_refinements(t))
                    .collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), self.resolve_refinements(t)))
                    .collect(),
                result: Box::new(self.resolve_refinements(result)),
            },
            Type::And(conjuncts) => Type::And(
                conjuncts
                    .iter()
                    .map(|t| self.resolve_refinements(t))
                    .collect(),
            ),
            Type::Tuple(items) => {
                Type::Tuple(items.iter().map(|t| self.resolve_refinements(t)).collect())
            }
            Type::List(item) => Type::List(Box::new(self.resolve_refinements(item))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), self.resolve_refinements(t)))
                    .collect(),
            ),
            Type::Forall(vars, body) => {
                Type::Forall(vars.clone(), Box::new(self.resolve_refinements(body)))
            }
            _ => ty.clone(),
        }
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

    /// Renders `ty` for an error message, with meta solutions applied and
    /// refinement variables resolved to the sorts they may inhabit.
    fn display(&self, ty: &Type) -> String {
        self.resolve_refinements(&ty.apply(&self.subst)).to_string()
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
        // Refinement variables local to the type freeze with variance:
        // covariant positions to the join of their guarantees, contravariant
        // (parameter) positions to the meet of their contracts (the
        // variance-directed endpoint choice of Pierce and Turner's local
        // type inference); no information means "any numeric". Variables
        // still reachable from the context (an outer parameter's) stay
        // live, mirroring the meta exclusion below. There is no bounded
        // quantification.
        let mut context_refinements = Vec::new();
        for (_, entry) in context.iter() {
            if let ContextEntry::Ty(ty) = entry {
                ty.free_refinements(&self.subst, &mut context_refinements);
            }
        }
        let applied = self.freeze_refinements(&applied, true, &context_refinements);
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

    /// Freezes the refinement variables of `ty` at generalization time,
    /// with `positive` tracking variance (parameters flip it). Variables in
    /// `keep` stay live.
    fn freeze_refinements(&mut self, ty: &Type, positive: bool, keep: &[u32]) -> Type {
        match ty {
            Type::Numeric(Refinement::Var(id)) if !keep.contains(id) => {
                let var = &self.refs[*id as usize];
                let sort = if positive {
                    if !var.lower.is_empty() {
                        var.lower
                    } else {
                        Sort::TOP
                    }
                } else {
                    var.upper.unwrap_or(Sort::TOP)
                };
                Type::ground(sort)
            }
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional
                    .iter()
                    .map(|t| self.freeze_refinements(t, !positive, keep))
                    .collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), self.freeze_refinements(t, !positive, keep)))
                    .collect(),
                result: Box::new(self.freeze_refinements(result, positive, keep)),
            },
            Type::And(conjuncts) => Type::And(
                conjuncts
                    .iter()
                    .map(|t| self.freeze_refinements(t, positive, keep))
                    .collect(),
            ),
            Type::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|t| self.freeze_refinements(t, positive, keep))
                    .collect(),
            ),
            Type::List(item) => Type::List(Box::new(self.freeze_refinements(item, positive, keep))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), self.freeze_refinements(t, positive, keep)))
                    .collect(),
            ),
            Type::Forall(vars, body) => Type::Forall(
                vars.clone(),
                Box::new(self.freeze_refinements(body, positive, keep)),
            ),
            _ => ty.clone(),
        }
    }

    /// Infers the type of a definition's right-hand side: the ordinary
    /// inference pass, plus tabulation into an intersection of arrows for
    /// the functions `tabulate` covers. Callers generalize the result as
    /// they would any inferred type.
    fn infer_definition<M>(&mut self, context: &mut TypeContext, expr: &SourceExpr<M, S>) -> Type {
        let base = self.infer(context, &mut Vec::new(), expr);
        self.tabulate(context, expr, &base).unwrap_or(base)
    }

    /// Tabulates a definition-bound function over its numeric parameters —
    /// Freeman and Pfenning §4: the principal refinement type of a
    /// definition is a finite intersection of arrows, found by re-checking
    /// the body at each point of the finite refinement lattice, here each
    /// vector of atoms over the numeric parameters (positional and named
    /// alike). Structural parameters ride along as fresh unknowns (solved
    /// per row by the body, quantified by the caller's generalization),
    /// and a structural named parameter's default flows in as a guarantee,
    /// as in the base pass. A vector whose body check errors contributes no
    /// row (the function is not applicable there), and the exploratory
    /// errors are discarded — the base pass has already reported
    /// anything unconditional.
    ///
    /// Beyond `MAX_TABULATION_VECTORS`, enumeration retries on the
    /// two-point unseq/seq split — keeping the relational seq holes, the
    /// soundness-critical part, at the cost of the int/float distinctions
    /// — and returns `None` (keep the base type, the freeze-at-generalize
    /// summary) only past that, or for functions with no numeric
    /// parameters.
    ///
    /// This is what makes parameter contracts *relational* rather than
    /// per-position: `fn(a, b) => a + b` gets no `(seq, seq)` row, so a
    /// two-seq call errors even though each position separately admits a
    /// seq.
    fn tabulate<M>(
        &mut self,
        context: &mut TypeContext,
        expr: &SourceExpr<M, S>,
        base: &Type,
    ) -> Option<Type> {
        let Expr::Function {
            positional,
            named,
            body,
        } = &expr.expr
        else {
            return None;
        };
        if positional.is_empty() && named.is_empty() {
            return None;
        }
        let Type::Function {
            positional: parameters,
            named: named_parameters,
            ..
        } = self.resolve(base)
        else {
            return None;
        };
        if parameters.len() != positional.len() || named_parameters.len() != named.len() {
            return None;
        }
        let numeric: Vec<bool> = parameters
            .iter()
            .map(|parameter| matches!(self.resolve(parameter), Type::Numeric(_)))
            .collect();
        let named_numeric: Vec<bool> = named_parameters
            .iter()
            .map(|(_, parameter)| matches!(self.resolve(parameter), Type::Numeric(_)))
            .collect();
        let tabulated = numeric
            .iter()
            .chain(&named_numeric)
            .filter(|numeric| **numeric)
            .count();
        if tabulated == 0 {
            return None;
        }
        let full: &[Sort] = &[
            Sort::INT,
            Sort::NON_INT_ONLY,
            Sort::NON_CONST_WAVE,
            Sort::SEQ,
        ];
        let coarse: &[Sort] = &[Sort::WAVE, Sort::SEQ];
        let fits = |set: &[Sort]| {
            set.len()
                .checked_pow(u32::try_from(tabulated).ok()?)
                .filter(|vectors| *vectors <= MAX_TABULATION_VECTORS)
        };
        let (atoms, vectors) = if let Some(vectors) = fits(full) {
            (full, vectors)
        } else if let Some(vectors) = fits(coarse) {
            (coarse, vectors)
        } else {
            return None;
        };
        // Refinement variables reachable from the enclosing context belong
        // to outer scopes and must stay live in the rows (mirroring
        // `generalize`'s exclusion).
        let mut keep = Vec::new();
        for (_, entry) in context.iter() {
            if let ContextEntry::Ty(ty) = entry {
                ty.free_refinements(&self.subst, &mut keep);
            }
        }
        let mut rows = Vec::new();
        for index in 0..vectors {
            let errors = self.errors.len();
            let mark = self.mark();
            let depth = context.len();
            let mut cursor = 0u32;
            let mut domains = Vec::with_capacity(positional.len());
            for (pattern, numeric) in positional.iter().zip(&numeric) {
                if *numeric {
                    let atom = atoms[(index / atoms.len().pow(cursor)) % atoms.len()];
                    cursor += 1;
                    self.bind_pattern(context, pattern, Type::ground(atom), &expr.span, false);
                    domains.push(Type::ground(atom));
                } else {
                    domains.push(self.pattern_param_type(context, pattern));
                }
            }
            let mut named_domains = Vec::with_capacity(named.len());
            for ((name, default), numeric) in named.iter().zip(&named_numeric) {
                let parameter = if *numeric {
                    let atom = atoms[(index / atoms.len().pow(cursor)) % atoms.len()];
                    cursor += 1;
                    Type::ground(atom)
                } else {
                    let default_ty = self.infer(context, &mut Vec::new(), default);
                    let parameter = self.fresh_meta();
                    self.subtype_check(&default_ty, &parameter, &default.span);
                    parameter
                };
                context.push((name.clone(), ContextEntry::Ty(parameter.clone())));
                named_domains.push((name.clone(), parameter));
            }
            let result = self.infer(context, &mut Vec::new(), body);
            context.truncate(depth);
            if self.errors.len() == errors {
                // Resolve the row fully before rolling back the state it
                // was solved in.
                let row = Type::Function {
                    positional: domains,
                    named: named_domains,
                    result: Box::new(result),
                }
                .apply(&self.subst);
                rows.push(self.freeze_refinements(&row, true, &keep));
            }
            self.errors.truncate(errors);
            self.rollback(mark);
        }
        if rows.is_empty() {
            return None;
        }
        let mut rows = merge_rows(rows);
        Some(if rows.len() == 1 {
            rows.remove(0)
        } else {
            Type::And(rows)
        })
    }

    /// Unifies two types, solving metas by equality — Fig. 13. Fails
    /// silently; callers decide whether a failure warrants an error.
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
            // A meta meeting a numeric solves to a fresh refinement
            // variable, so a `∀a` instantiation becomes refinement-joining
            // at its first numeric binding.
            (Type::Meta(id), Type::Numeric(rep)) | (Type::Numeric(rep), Type::Meta(id)) => {
                let fresh = self.fresh_refinement();
                self.solve(*id, Type::Numeric(fresh.clone()));
                self.numeric_unify(&fresh, rep)
            }
            // AU-Var1/AU-Var2: solve an unsolved meta, under the occurs
            // check (`α̂ ∉ ftv(S τ)`). AU-BVar1/AU-BVar2 (already-solved
            // metas) are handled by `resolve` above.
            (Type::Meta(id), other) | (other, Type::Meta(id)) => {
                let mut metas = Vec::new();
                other.free_metas(&self.subst, &mut metas);
                if metas.contains(id) {
                    return Err(());
                }
                self.solve(*id, other.clone());
                Ok(())
            }
            // Numerics merge their bounds, failing only when one side's
            // guarantees conflict with the other's recorded contracts.
            (Type::Numeric(x), Type::Numeric(y)) => self.numeric_unify(x, y),
            (Type::Var(x), Type::Var(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err(())
                }
            }
            // AU-Refl, extended to tuun's base types.
            (Type::Bool, Type::Bool) | (Type::String, Type::String) => Ok(()),
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
    /// subtyping judgment of Fig. 15, extended with tuun's numeric sorts,
    /// intersections of arrows, and `Dynamic`.
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
            // Containment judgment plus guarantee and contract flows;
            // see `numeric_subtype`.
            (Type::Numeric(found), Type::Numeric(expected)) => {
                self.numeric_subtype(found, expected)
            }
            // A meta meeting a numeric in a subtype position becomes a
            // refinement variable with the flow direction preserved: an
            // expected numeric imposes its contract, a found numeric
            // records its guarantee. (Falling through to `unify` would
            // erase the direction.)
            (Type::Meta(id), Type::Numeric(expected)) => {
                let fresh = self.fresh_refinement();
                self.solve(*id, Type::Numeric(fresh.clone()));
                self.numeric_subtype(&fresh, expected)
            }
            (Type::Numeric(found), Type::Meta(id)) => {
                let fresh = self.fresh_refinement();
                self.solve(*id, Type::Numeric(fresh.clone()));
                self.numeric_subtype(found, &fresh)
            }
            // An intersection against an expected arrow selects with the
            // arrow's parameters as a pseudo-frame — the same applicative
            // subtyping as at applications, so committing to one conjunct
            // cannot poison a sibling argument: when no single row applies
            // outright, the coverage join plays the summary role
            // (⋀(Aᵢ→Bᵢ) <: (⋃Aᵢ)→(⋁Bᵢ), sound by atom disjointness).
            (
                Type::And(conjuncts),
                Type::Function {
                    positional,
                    named,
                    result,
                },
            ) if named.is_empty() => {
                // Selecting against unsolved variables iterates to a fixed
                // point (Freeman's abstract interpretation): when the arrow's
                // result flows back into a parameter variable — `unfold`'s `a →
                // a`, `reduce`'s accumulator — the growth can change which rows
                // apply and what contracts they may impose, so a one-shot
                // selection over-commits (e.g. a float seed selects the float
                // row, whose result grows the variable and then violates the
                // row's own contract). Roll the attempt back, carry the growth,
                // and reselect; sorts only grow, so this terminates within the
                // lattice height.
                let variables: Vec<u32> = positional
                    .iter()
                    .filter_map(|parameter| match self.resolve(parameter) {
                        Type::Numeric(Refinement::Var(id)) => Some(id),
                        _ => None,
                    })
                    .collect();
                let mut iterations = 0;
                loop {
                    let before: Vec<Sort> = variables
                        .iter()
                        .map(|id| self.refs[*id as usize].lower)
                        .collect();
                    let mark = self.mark();
                    let pseudo = Frame {
                        positional: positional
                            .iter()
                            .map(|parameter| (parameter.clone(), None))
                            .collect(),
                        named: Vec::new(),
                        span: None,
                    };
                    let outcome = match self.select_core(conjuncts, &pseudo) {
                        Selection::Selected(row_result) => self.subtype(&row_result, result),
                        Selection::NoApplicableRow => Err(()),
                        Selection::NoMatchingRows => {
                            self.subtype_any_conjunct(conjuncts.clone(), &b)
                        }
                    };
                    if outcome.is_err() {
                        break outcome;
                    }
                    let after: Vec<Sort> = variables
                        .iter()
                        .map(|id| self.refs[*id as usize].lower)
                        .collect();
                    iterations += 1;
                    if before == after || iterations >= 8 {
                        break outcome;
                    }
                    self.rollback(mark);
                    let mut carried = Ok(());
                    for (id, sort) in variables.iter().zip(after) {
                        if self.join_lower(*id, sort).is_err() {
                            carried = Err(());
                            break;
                        }
                    }
                    if carried.is_err() {
                        break carried;
                    }
                }
            }
            // An intersection is usable where any of its conjuncts is
            // (rules Sub-And-L/Sub-And-R); failed attempts roll back their
            // partial solving.
            //
            // Expected arrows *with named parameters* deliberately land
            // here too (the guard above): a pseudo-frame models a call,
            // and a promised named parameter is not a call argument — as
            // an "omitted argument" it would skip its contravariant check,
            // and a row lacking the name would wrongly stay eligible. The
            // per-conjunct path does the full named discipline per row,
            // with the first fitting row committing — no distributive
            // reading, no fixed-point iteration.
            // TODO extend the pseudo-frame with promised named parameters
            // (and filter rows to those offering them) so named-having
            // arrows join the selection path.
            (Type::And(conjuncts), _) => self.subtype_any_conjunct(conjuncts.clone(), &b),
            // Meeting an intersection requires meeting every conjunct
            // (rule Sub-And).
            (_, Type::And(conjuncts)) => {
                for conjunct in conjuncts.clone() {
                    self.subtype(&a, &conjunct)?;
                }
                Ok(())
            }
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

    /// Tries `<:` against each conjunct in order, keeping the first
    /// success and rolling back failed attempts (rules
    /// Sub-And-L/Sub-And-R).
    fn subtype_any_conjunct(&mut self, conjuncts: Vec<Type>, b: &Type) -> Result<(), ()> {
        for conjunct in conjuncts {
            let mark = self.mark();
            if self.subtype(&conjunct, b).is_ok() {
                return Ok(());
            }
            self.rollback(mark);
        }
        Err(())
    }

    /// Checks `found <: expected` and reports an error at `span` on
    /// failure, rolling back any partial solving from the failed attempt.
    fn subtype_check(&mut self, found: &Type, expected: &Type, span: &Option<Span<S>>) {
        let mark = self.mark();
        if self.subtype(found, expected).is_err() {
            self.rollback(mark);
            let message = format!(
                "expected {}, found {}",
                self.display(expected),
                self.display(found)
            );
            self.error(message, span);
        }
    }

    /// Returns the join of the two types (a tuun extension; the paper has no
    /// join because it has no subtyping between base types): unification if the
    /// types agree structurally, sort union for numerics (mixed
    /// float/waveform/seq lists and branches are common), pointwise for lists
    /// and tuples, and `Dynamic` with an error otherwise. Quantified types join
    /// at an instance.
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
            // Numerics join by sort union.
            (Type::Numeric(x), Type::Numeric(y)) => {
                Type::ground(self.may_of(x).union(self.may_of(y)))
            }
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
                let mark = self.mark();
                if self.unify(&a, &b).is_ok() {
                    self.resolve(&a)
                } else {
                    self.rollback(mark);
                    let message = format!(
                        "incompatible types {} and {}",
                        self.display(&a),
                        self.display(&b)
                    );
                    self.error(message, span);
                    Type::Dynamic
                }
            }
        }
    }

    /// Applies `ty` to the pending applications in `psi`, returning the
    /// residual result type — the application subtyping judgment
    /// `(S, N) ∣ Ψ ⊢ A <: B` of Fig. 15 (bottom), with one frame per call.
    fn app_subtype(&mut self, psi: &mut Psi<S>, ty: Type, head: Option<&str>) -> Type {
        // AS-Empty: not applied to anything, so the type stands as is.
        let Some(frame) = psi.pop() else { return ty };
        match self.resolve(&ty) {
            // AS-ForallL2: instantiate and keep consuming.
            Type::Forall(vars, body) => {
                let instance = self.instantiate(&vars, *body);
                psi.push(frame);
                self.app_subtype(psi, instance, head)
            }
            // AS-Fun2, n-ary: the frame's arguments feed one call; continue
            // with the rest of Ψ against the result type.
            Type::Function {
                positional,
                named,
                result,
            } => {
                self.check_frame(&frame, &positional, &named);
                self.app_subtype(psi, *result, head)
            }
            // Applicative selection from an intersection (Xue et al.):
            // conjuncts are tried in table order against the frame's
            // argument sorts; see `select_conjunct`.
            Type::And(conjuncts) => match self.select_conjunct(&conjuncts, &frame, head) {
                Some(result) => self.app_subtype(psi, result, head),
                None => {
                    psi.clear();
                    Type::Dynamic
                }
            },
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
                // Cannot fail: the meta is unsolved and the arrow is
                // built of fresh metas.
                let _ = self.unify(&meta, &function);
                psi.push(frame);
                self.app_subtype(psi, function, head)
            }
            Type::Dynamic => {
                psi.clear();
                Type::Dynamic
            }
            // Applying a non-function: the static form of evaluation's
            // "Invalid application" error.
            other => {
                let message = format!("cannot apply a value of type {}", self.display(&other));
                self.error(message, &frame.span);
                psi.clear();
                Type::Dynamic
            }
        }
    }

    /// Selects from an intersection of arrows against one frame — Freeman's
    /// `apptype`, Xue et al.'s applicative subtyping `A ≪ S` with the frame as
    /// the selector:
    ///
    /// - the first conjunct that applies outright supplies the result: each
    ///   numeric argument's sort contained in the domain, each structural
    ///   argument a subtype of it (table order is most-specific-first,
    ///   mirroring the runtime match arms);
    /// - otherwise coverage decides (`select_by_atoms`): every atom combination
    ///   of the arguments must have an accepting row, and the covering rows'
    ///   results join;
    /// - no coverage means the runtime has no matching arm: a rejection.
    ///
    /// Serves elimination and checking alike: `app_subtype` selects with a real
    /// Ψ frame, and `subtype` selects with a pseudo-frame built from an
    /// expected arrow's parameters.
    fn select_core(&mut self, conjuncts: &[Type], frame: &Frame<S>) -> Selection {
        // The argument sorts. Dynamic and unsolved metas may be any
        // numeric; a structural argument (list, function, ...) has no sort
        // and is checked against each row's domain by subtyping.
        let sorts: Vec<Option<Sort>> = frame
            .positional
            .iter()
            .map(|(argument, _)| match self.resolve(argument) {
                Type::Numeric(rep) => Some(self.may_of(&rep)),
                Type::Dynamic | Type::Meta(_) => Some(Sort::TOP),
                _ => None,
            })
            .collect();
        let rows: Vec<Type> = conjuncts
            .iter()
            .filter(|conjunct| {
                matches!(conjunct, Type::Function { positional, .. }
                    if positional.len() == frame.positional.len())
            })
            .cloned()
            .collect();
        if rows.is_empty() {
            return Selection::NoMatchingRows;
        }
        // Contracts for unsolved arguments use the union of every row's domain
        // at the position: the variable may still grow (an arrow's result can
        // feed back into it — see the fixed-point iteration in `subtype`), and
        // any narrower contract, such as a chosen row's own domain, would
        // reject that growth. Ground arguments are already judged by
        // applicability.
        let broad = self.broad_domains(&rows, frame.positional.len());
        // First definitely-applicable row wins; its structural subtyping
        // commits (and is rolled back when a later position rejects it).
        for row in &rows {
            let mark = self.mark();
            if self.row_applies(row, &sorts, frame).is_some() {
                self.record_selection_contracts(frame, &broad);
                let Type::Function { result, .. } = row else {
                    unreachable!("rows are arrows");
                };
                return Selection::Selected((**result).clone());
            }
            self.rollback(mark);
        }
        // Otherwise, atom-decomposition coverage.
        self.select_by_atoms(&rows, &sorts, frame)
    }

    /// The per-position union of every row's domain sort (⊤ where a row's
    /// domain is unknown or structural) — the loosest honest contract for
    /// an argument that may still grow.
    fn broad_domains(&self, rows: &[Type], arity: usize) -> Vec<Sort> {
        let mut broad = vec![Sort::NONE; arity];
        for row in rows {
            let Type::Function { positional, .. } = row else {
                unreachable!("rows are arrows");
            };
            for (broad, domain) in broad.iter_mut().zip(positional) {
                let sort = match self.resolve(domain) {
                    Type::Numeric(rep) => self.may_of(&rep),
                    _ => Sort::TOP,
                };
                *broad = broad.union(sort);
            }
        }
        broad
    }

    /// Returns the row's domains as contract sorts when the row applies
    /// outright — every argument's sort contained in its domain — and
    /// `None` otherwise. Numeric positions check by sort containment and
    /// report their domain sort; unknown and structural domains accept by
    /// subtyping and report ⊤ (no sort contract). A supplied named
    /// argument checks against the row's named domain; an omitted one
    /// takes the default, whose sort the rows do not record, so the row
    /// does not apply outright — coverage (`select_by_atoms`) is the path
    /// that accepts it. Structural subtyping solves state, so callers
    /// snapshot around the call.
    fn row_applies(
        &mut self,
        row: &Type,
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Option<Vec<Sort>> {
        let Type::Function {
            positional, named, ..
        } = row
        else {
            unreachable!("rows are arrows");
        };
        let mut domains = Vec::with_capacity(positional.len());
        for ((sort, domain), (argument, _)) in sorts.iter().zip(positional).zip(&frame.positional) {
            let (fits, domain) = self.position_fits(argument, *sort, domain);
            domains.push(domain);
            if !fits {
                return None;
            }
        }
        for (name, domain) in named {
            match frame.named.iter().find(|(n, _, _)| n == name) {
                Some((_, argument, _)) => {
                    let sort = match self.resolve(argument) {
                        Type::Numeric(rep) => Some(self.may_of(&rep)),
                        Type::Dynamic | Type::Meta(_) => Some(Sort::TOP),
                        _ => None,
                    };
                    // TODO record named-argument contracts the way
                    // `record_selection_contracts` does for positional ones.
                    let (fits, _) = self.position_fits(argument, sort, domain);
                    if !fits {
                        return None;
                    }
                }
                // An omitted argument takes the default, whose sort the
                // rows do not record; coverage (`select_by_atoms`) is the
                // path that can accept it.
                None => return None,
            }
        }
        Some(domains)
    }

    /// Selection by atom coverage (sorts are unions over a disjoint atom
    /// basis — Freeman's union normal form): a runtime value inhabits
    /// exactly one atom, so a call is covered when every combination of
    /// its arguments' atoms has a row that accepts it; the result is the
    /// join of the covering rows' results, and each argument's contract is
    /// the union of the domains that admitted it. A combination no row
    /// covers means the runtime has no matching arm: a rejection.
    ///
    /// Ground numeric arguments decompose into their atoms. Unsolved
    /// refinement variables and metas defer — they pass here and their
    /// judgment happens where their bounds are final — as does `Dynamic`
    /// (the recovery type passes everything, imposing nothing).
    /// Structural arguments constrain per row by subtyping.
    fn select_by_atoms(
        &mut self,
        rows: &[Type],
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Selection {
        // Per-position atom choices; `None` is a wildcard (deferred or
        // structural, constrained elsewhere). An unsolved variable's known
        // guarantees are judged now — lower bounds only grow, so an
        // uncovered lower atom is a final violation — while its future
        // atoms stay deferred to its recorded contracts.
        let choices: Vec<Option<Vec<Sort>>> = sorts
            .iter()
            .zip(&frame.positional)
            .map(
                |(sort, (argument, _))| match (sort, self.resolve(argument)) {
                    (Some(sort), Type::Numeric(Refinement::Ground(_))) => {
                        Some(sort.atoms().collect())
                    }
                    (_, Type::Numeric(Refinement::Var(id))) => {
                        let lower = self.refs[id as usize].lower;
                        if lower.is_empty() {
                            None
                        } else {
                            Some(lower.atoms().collect())
                        }
                    }
                    _ => None,
                },
            )
            .collect();
        // Per row: domain sorts (⊤ for unknown/structural domains), the
        // structural and named constraints, and the result resolved before
        // the probe's rollback.
        struct Candidate {
            domains: Vec<Sort>,
            result: Type,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for row in rows {
            let mark = self.mark();
            let applies = self.row_admits(row, sorts, frame);
            let candidate = applies.map(|domains| {
                let Type::Function { result, .. } = row else {
                    unreachable!("rows are arrows");
                };
                Candidate {
                    domains,
                    result: result.apply(&self.subst),
                }
            });
            self.rollback(mark);
            if let Some(candidate) = candidate {
                candidates.push(candidate);
            }
        }
        if candidates.is_empty() {
            return Selection::NoApplicableRow;
        }
        // Enumerate the atom combinations (the empty product is one empty
        // combination); each must be covered by some candidate.
        let broad = self.broad_domains(rows, frame.positional.len());
        let mut result: Option<Type> = None;
        let mut odometer: Vec<usize> = choices.iter().map(|_| 0).collect();
        loop {
            let covering = candidates.iter().find(|candidate| {
                choices.iter().zip(&odometer).zip(&candidate.domains).all(
                    |((choice, digit), domain)| match choice {
                        Some(atoms) => atoms[*digit].is_subset(*domain),
                        None => true,
                    },
                )
            });
            let Some(covering) = covering else {
                return Selection::NoApplicableRow;
            };
            result = Some(match result {
                None => covering.result.clone(),
                Some(previous) => self.join(previous, covering.result.clone(), &frame.span),
            });
            // Advance the odometer; done when it wraps.
            let mut position = 0;
            loop {
                if position == odometer.len() {
                    let result = result.expect("at least one combination");
                    self.record_selection_contracts(frame, &broad);
                    return Selection::Selected(result);
                }
                match &choices[position] {
                    Some(atoms) if odometer[position] + 1 < atoms.len() => {
                        odometer[position] += 1;
                        break;
                    }
                    _ => {
                        odometer[position] = 0;
                        position += 1;
                    }
                }
            }
        }
    }

    /// Whether a row can participate in coverage at all: structural arguments
    /// must subtype their domains, supplied named arguments must fit theirs
    /// (omitted ones defer), and each position's contract sort is reported as
    /// in `row_applies`. Numeric positions are not judged here — coverage
    /// judges them atom by atom.
    fn row_admits(
        &mut self,
        row: &Type,
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Option<Vec<Sort>> {
        let Type::Function {
            positional, named, ..
        } = row
        else {
            unreachable!("rows are arrows");
        };
        let mut domains = Vec::with_capacity(positional.len());
        for ((sort, domain), (argument, _)) in sorts.iter().zip(positional).zip(&frame.positional) {
            let resolved = self.resolve(domain);
            let fits = match (sort, &resolved) {
                // Numeric against numeric is coverage's job.
                (Some(_), Type::Numeric(rep)) => {
                    domains.push(self.may_of(rep));
                    true
                }
                (None, Type::Numeric(_)) => false,
                (_, Type::Meta(_) | Type::Var(_) | Type::Dynamic) => {
                    domains.push(Sort::TOP);
                    true
                }
                (Some(_), _) => false,
                (None, _) => {
                    domains.push(Sort::TOP);
                    self.subtype(argument, &resolved).is_ok()
                }
            };
            if !fits {
                return None;
            }
        }
        for (name, domain) in named {
            if let Some((_, argument, _)) = frame.named.iter().find(|(n, _, _)| n == name) {
                let sort = match self.resolve(argument) {
                    Type::Numeric(rep) => Some(self.may_of(&rep)),
                    Type::Dynamic | Type::Meta(_) => Some(Sort::TOP),
                    _ => None,
                };
                let (fits, _) = self.position_fits(argument, sort, domain);
                if !fits {
                    return None;
                }
            }
        }
        Some(domains)
    }

    /// Whether one argument fits one domain — sort containment for
    /// numerics, subtyping for structural arguments — and the contract
    /// sort the domain imposes (⊤ for unknown and structural domains).
    /// Structural subtyping solves state; callers snapshot.
    fn position_fits(
        &mut self,
        argument: &Type,
        sort: Option<Sort>,
        domain: &Type,
    ) -> (bool, Sort) {
        let resolved = self.resolve(domain);
        match (sort, &resolved) {
            (Some(sort), Type::Numeric(rep)) => {
                let domain = self.may_of(rep);
                (sort.is_subset(domain), domain)
            }
            // A structural argument never fits a numeric domain.
            (None, Type::Numeric(_)) => (false, Sort::TOP),
            // An unconstrained domain accepts any argument without
            // imposing a sort contract.
            (_, Type::Meta(_) | Type::Var(_) | Type::Dynamic) => (true, Sort::TOP),
            // A numeric argument has no arm at a structural domain.
            (Some(_), _) => (false, Sort::TOP),
            (None, _) => (self.subtype(argument, &resolved).is_ok(), Sort::TOP),
        }
    }

    /// Selects with `select_core` and reports an error when nothing
    /// applies; returns `None` after reporting.
    fn select_conjunct(
        &mut self,
        conjuncts: &[Type],
        frame: &Frame<S>,
        head: Option<&str>,
    ) -> Option<Type> {
        // A named argument that no conjunct declares can never select
        // anything.
        for (name, _, _) in &frame.named {
            let declared = conjuncts.iter().any(|conjunct| {
                matches!(conjunct, Type::Function { named, .. }
                    if named.iter().any(|(n, _)| n == name))
            });
            if !declared {
                self.error(format!("no named parameter \"{}\"", name), &frame.span);
            }
        }
        match self.select_core(conjuncts, frame) {
            Selection::Selected(result) => Some(result),
            // Nothing has the call's shape: when the table is unambiguous
            // about its arity, report the arity mismatch the way
            // `check_frame` would.
            Selection::NoMatchingRows => {
                let mut arities: Vec<&Vec<Type>> = Vec::new();
                for conjunct in conjuncts {
                    if let Type::Function { positional, .. } = conjunct
                        && !arities.iter().any(|known| known.len() == positional.len())
                    {
                        arities.push(positional);
                    }
                }
                let message = match &arities[..] {
                    [parameters] if frame.positional.len() > parameters.len() => {
                        "extra positional parameter".to_string()
                    }
                    [parameters] if frame.positional.len() < parameters.len() => {
                        format!(
                            "missing parameter of type {}",
                            self.display(&parameters[frame.positional.len()])
                        )
                    }
                    _ => self.no_use_message(frame, head),
                };
                self.error(message, &frame.span);
                None
            }
            Selection::NoApplicableRow => {
                // When one argument alone rules out every row, pinpoint it
                // the way `check_frame` would: report at that argument with
                // the union of the domains it failed.
                let rows: Vec<&Type> = conjuncts
                    .iter()
                    .filter(|conjunct| {
                        matches!(conjunct, Type::Function { positional, .. }
                            if positional.len() == frame.positional.len())
                    })
                    .collect();
                for (position, (argument, span)) in frame.positional.iter().enumerate() {
                    let Type::Numeric(rep) = self.resolve(argument) else {
                        continue;
                    };
                    let may = self.may_of(&rep);
                    let mut union = Sort::NONE;
                    let mut sorted = true;
                    for row in &rows {
                        let Type::Function { positional, .. } = row else {
                            unreachable!("rows are arrows");
                        };
                        match self.resolve(&positional[position]) {
                            Type::Numeric(rep) => union = union.union(self.may_of(&rep)),
                            _ => sorted = false,
                        }
                    }
                    if sorted && !rows.is_empty() && may.intersect(union).is_empty() {
                        let message = format!(
                            "expected {}, found {}",
                            Type::ground(union),
                            self.display(argument)
                        );
                        self.error(message, span);
                        return None;
                    }
                }
                let mut seqs = 0;
                for (argument, _) in &frame.positional {
                    if let Type::Numeric(rep) = self.resolve(argument)
                        && self.may_of(&rep) == Sort::SEQ
                    {
                        seqs += 1;
                    }
                }
                let both_seqs = frame.positional.len() == 2 && seqs == 2;
                let message = match (both_seqs, head) {
                    (true, Some(name)) => format!("cannot combine two seqs with {}", name),
                    (true, None) => "cannot combine two seqs".to_string(),
                    (false, name) => self.no_use_message(frame, name),
                };
                self.error(message, &frame.span);
                None
            }
        }
    }

    /// The "no use of f accepts (...)" message, listing the call's
    /// positional and named arguments.
    fn no_use_message(&self, frame: &Frame<S>, head: Option<&str>) -> String {
        let arguments = frame
            .positional
            .iter()
            .map(|(ty, _)| self.display(ty))
            .chain(
                frame
                    .named
                    .iter()
                    .map(|(name, ty, _)| format!("{} = {}", name, self.display(ty))),
            )
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "no use of {} accepts ({})",
            head.unwrap_or("this function"),
            arguments
        )
    }

    /// Bounds the frame's variable arguments by the domains that admitted
    /// them, so a parameter used with an operator inherits the operator's
    /// requirements.
    fn record_selection_contracts(&mut self, frame: &Frame<S>, domains: &[Sort]) {
        for ((argument, span), domain) in frame.positional.iter().zip(domains) {
            let found = match self.resolve(argument) {
                Type::Numeric(rep) => rep,
                // An unconstrained argument commits to numeric here — the
                // table's rows are all the runtime arms there are — so it
                // becomes a refinement variable under the same contract
                // (mirroring `subtype`'s meta-meets-numeric arms).
                Type::Meta(meta) => {
                    let fresh = self.fresh_refinement();
                    self.solve(meta, Type::Numeric(fresh.clone()));
                    fresh
                }
                _ => continue,
            };
            // A contract contradicting the argument's bounds — one the
            // applicability sorts could not see — errors at that argument.
            if self
                .numeric_subtype(&found, &Refinement::Ground(*domain))
                .is_err()
            {
                let message = format!(
                    "expected {}, found {}",
                    Type::ground(*domain),
                    self.display(argument)
                );
                self.error(message, span);
            }
        }
    }

    /// Checks one call's arguments against a function type's parameters:
    /// arity, named-parameter existence, and argument subtyping (the
    /// premises of rule AS-Fun2, plus tuun's arity and named-argument checks,
    /// which evaluation performs at application time).
    fn check_frame(&mut self, frame: &Frame<S>, positional: &[Type], named: &[(String, Type)]) {
        if frame.positional.len() > positional.len() {
            self.error("extra positional parameter".to_string(), &frame.span);
        } else if frame.positional.len() < positional.len() {
            let missing = self.display(&positional[frame.positional.len()]);
            self.error(
                format!("missing parameter of type {}", missing),
                &frame.span,
            );
        }
        // Arguments carrying tabulated tables (intersections) check after
        // the rest, so row selection sees the sibling arguments' flows
        // first — e.g. a fold's seed constrains the accumulator before
        // the fold function's row is chosen.
        let table = |infer: &Self, ty: &Type| match infer.resolve(ty) {
            Type::And(_) => true,
            Type::Forall(_, body) => matches!(*body, Type::And(_)),
            _ => false,
        };
        for ((argument, span), parameter) in frame.positional.iter().zip(positional) {
            if !table(self, argument) {
                self.subtype_check(argument, parameter, span);
            }
        }
        for ((argument, span), parameter) in frame.positional.iter().zip(positional) {
            if table(self, argument) {
                self.subtype_check(argument, parameter, span);
            }
        }
        for (name, argument, span) in &frame.named {
            match named.iter().find(|(n, _)| n == name) {
                Some((_, parameter)) => {
                    let parameter = parameter.clone();
                    self.subtype_check(argument, &parameter, span);
                }
                None => self.error(format!("no named parameter \"{}\"", name), &frame.span),
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
            Expr::Bool(_) => self.app_subtype(psi, Type::Bool, None),
            Expr::String(_) => self.app_subtype(psi, Type::String, None),
            // A constant waveform is a number literal, refined by its
            // integrality; anything else waveform-valued is definitely a
            // (plain) waveform.
            Expr::Waveform(waveform) => {
                let ty = match waveform {
                    waveform::Waveform::Const(value) => {
                        if value.fract() == 0.0 {
                            Type::int()
                        } else {
                            Type::ground(Sort::NON_INT_ONLY)
                        }
                    }
                    _ => Type::non_const_wave(),
                };
                self.app_subtype(psi, ty, None)
            }
            Expr::Seq { offset, waveform } => {
                let offset_ty = self.infer(context, &mut Vec::new(), offset);
                self.subtype_check(&offset_ty, &Type::waveform(), &offset.span);
                let waveform_ty = self.infer(context, &mut Vec::new(), waveform);
                self.subtype_check(&waveform_ty, &Type::waveform(), &waveform.span);
                self.app_subtype(psi, Type::seq(), None)
            }
            // AT-Var: look the variable up and apply its type to Ψ.
            Expr::Variable(name) => match context.iter().rev().find(|(n, _)| n == name) {
                Some((_, ContextEntry::Ty(ty))) => {
                    let ty = ty.clone();
                    self.app_subtype(psi, ty, Some(name))
                }
                Some((_, ContextEntry::Builtin(builtin))) => {
                    let builtin = builtin.clone();
                    let ty = signatures::signature(&builtin).unwrap_or(Type::Dynamic);
                    self.app_subtype(psi, ty, Some(&builtin))
                }
                None => {
                    self.error(format!("unbound variable '{}'", name), &expr.span);
                    psi.clear();
                    Type::Dynamic
                }
            },
            // AT-Var for built-ins: the signature table stands in for the
            // typing-context entry.
            Expr::BuiltIn { name, .. } => {
                let ty = signatures::signature(name).unwrap_or(Type::Dynamic);
                self.app_subtype(psi, ty, Some(name))
            }
            Expr::Function {
                positional,
                named,
                body,
            } => {
                // A named parameter's type comes from the body's use of
                // it, not from its default: the default is only the value
                // the parameter takes when a call omits the argument, so
                // its type (inferred in the enclosing scope, mirroring
                // once-at-definition evaluation) flows in as a guarantee
                // against the parameter's unknown. Not in the paper, which
                // has no named parameters.
                let named_types: Vec<(String, Type)> = named
                    .iter()
                    .map(|(name, default)| {
                        let default_ty = self.infer(context, &mut Vec::new(), default);
                        let parameter = self.fresh_meta();
                        self.subtype_check(&default_ty, &parameter, &default.span);
                        (name.clone(), parameter)
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
                            self.error("extra positional parameter".to_string(), &frame.span);
                        } else if frame.positional.len() < positional.len() {
                            let missing = &positional[frame.positional.len()];
                            self.error(format!("missing parameter \"{}\"", missing), &frame.span);
                        }
                        for (index, pattern) in positional.iter().enumerate() {
                            let (ty, span) = if arity_matches {
                                frame.positional[index].clone()
                            } else {
                                // Don't let an arity mismatch cascade into
                                // per-argument mismatch errors.
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
                                    self.error(
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
                        let ty = self.infer_definition(context, argument);
                        (self.generalize(context, ty), argument.span.clone())
                    })
                    .collect();
                let named_types = named
                    .iter()
                    .map(|(name, argument)| {
                        let ty = self.infer_definition(context, argument);
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
                self.app_subtype(psi, joined, None)
            }
            // Pairs take the inference-mode rule (the paper's Pair-I, §2.1):
            // tuples cannot be applied, so components are inferred with an
            // empty Ψ.
            Expr::Tuple(items) => {
                let types = items
                    .iter()
                    .map(|item| self.infer(context, &mut Vec::new(), item))
                    .collect();
                self.app_subtype(psi, Type::Tuple(types), None)
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
                self.app_subtype(psi, Type::List(Box::new(element)), None)
            }
            Expr::Project { module, name } => {
                let module_ty = self.infer(context, &mut Vec::new(), module);
                let ty = match self.resolve(&module_ty) {
                    Type::Module(entries) => match entries.iter().find(|(n, _)| n == name) {
                        Some((_, ty)) => ty.clone(),
                        None => {
                            self.error(format!("Module has no binding '{}'", name), &expr.span);
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
                        self.error(message, &expr.span);
                        Type::Dynamic
                    }
                };
                self.app_subtype(psi, ty, None)
            }
            Expr::BoundModule(entries) => {
                let types = entries
                    .iter()
                    .map(|(name, entry)| {
                        (name.clone(), self.infer(context, &mut Vec::new(), entry))
                    })
                    .collect();
                self.app_subtype(psi, Type::Module(types), None)
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
                    let tuple = Type::Tuple(patterns.iter().map(|_| self.fresh_meta()).collect());
                    // Cannot fail: the meta is unsolved and the tuple is
                    // built of fresh metas.
                    let _ = self.unify(&meta, &tuple);
                    self.bind_pattern(context, pattern, tuple, span, generalize_leaves);
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
                    self.error(message, span);
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
    /// An unresolvable module produces an error at its binding; its names
    /// then error as unbound wherever they are used.
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
                        self.error(message, &source_binding.span);
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
                            self.error(message, &source_binding.span);
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
                                    signatures::signature(&builtin).unwrap_or(Type::Dynamic)
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
                        let ty = self.infer_definition(context, expr);
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

/// Coalesces tabulated rows: two rows that differ at just one numeric domain
/// position and agree on the result merge into one row with the union domain
/// there — e.g. `({I}) -> float ∧ ({NonInt}) -> float` becomes `(float) ->
/// float`. Purely a simplification: selection reads the merged table the same
/// way, and displays stay legible.
fn merge_rows(mut rows: Vec<Type>) -> Vec<Type> {
    'restart: loop {
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                if let Some(merged) = merge_pair(&rows[i], &rows[j]) {
                    rows[i] = merged;
                    rows.remove(j);
                    continue 'restart;
                }
            }
        }
        return rows;
    }
}

/// Returns the union of rows `a` and `b` when they agree everywhere except at
/// most one numeric ground domain position (positional or named).
fn merge_pair(a: &Type, b: &Type) -> Option<Type> {
    let (
        Type::Function {
            positional: positional_a,
            named: named_a,
            result: result_a,
        },
        Type::Function {
            positional: positional_b,
            named: named_b,
            result: result_b,
        },
    ) = (a, b)
    else {
        return None;
    };
    if positional_a.len() != positional_b.len()
        || named_a.len() != named_b.len()
        || result_a != result_b
    {
        return None;
    }
    if named_a
        .iter()
        .zip(named_b)
        .any(|((name_a, _), (name_b, _))| name_a != name_b)
    {
        return None;
    }
    let mut merged = positional_a.clone();
    let mut merged_named = named_a.clone();
    let mut differences = 0;
    let union = |x: &Type, y: &Type, differences: &mut i32| -> Option<Option<Type>> {
        if x == y {
            return Some(None);
        }
        let (Type::Numeric(Refinement::Ground(sort_x)), Type::Numeric(Refinement::Ground(sort_y))) =
            (x, y)
        else {
            return None;
        };
        *differences += 1;
        if *differences > 1 {
            return None;
        }
        Some(Some(Type::ground(sort_x.union(*sort_y))))
    };
    for (position, (x, y)) in positional_a.iter().zip(positional_b).enumerate() {
        if let Some(unioned) = union(x, y, &mut differences)? {
            merged[position] = unioned;
        }
    }
    for (position, ((_, x), (_, y))) in named_a.iter().zip(named_b).enumerate() {
        if let Some(unioned) = union(x, y, &mut differences)? {
            merged_named[position].1 = unioned;
        }
    }
    Some(Type::Function {
        positional: merged,
        named: merged_named,
        result: result_a.clone(),
    })
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
    use crate::eval;
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

    fn messages(errors: &[Error<()>]) -> Vec<String> {
        errors.iter().map(|w| w.message().to_string()).collect()
    }

    #[track_caller]
    fn assert_clean(input: &str) {
        let errors = check(input);
        assert!(
            errors.is_empty(),
            "expected no errors for {:?}, got {:?}",
            input,
            messages(&errors)
        );
    }

    #[track_caller]
    fn assert_errors(input: &str, expected: &[&str]) {
        let errors = check(input);
        assert_eq!(
            messages(&errors),
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
    // errors.
    #[test]
    fn inference_mode_lambda_is_monomorphic() {
        assert_errors(
            "map(fn(g) => (g(1), g(\"s\")), [fn(x) => x])",
            &["expected int, found string"],
        );
    }

    // A lambda argument's table meets map's ∀ instantiation: the list
    // checks first (`check_frame` defers table-typed arguments), so the
    // element variable already holds {I} when the table is selected
    // against `(a) -> b`, and the int row covers it.
    #[test]
    fn map_over_a_lambda_argument() {
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

    // A mismatched argument errors and the error's span points at the
    // offending argument, not the whole call.
    #[test]
    fn argument_mismatch_points_at_argument() {
        let input = "sine(\"a\", 0)";
        let errors = check(input);
        assert_eq!(messages(&errors), ["expected waveform, found string"]);
        let start = input.find("\"a\"").unwrap();
        assert_eq!(errors[0].range(), Some(start..start + 3));
    }

    #[test]
    fn tuple_patterns() {
        assert_clean("let (a, b) = (1, \"x\") in a");
        // Two separate arguments do not destructure into a tuple pattern;
        // the arity error alone reports it (no per-argument cascade).
        assert_errors("(fn((y, z)) => y)(4, 5)", &["extra positional parameter"]);
    }

    // Named parameters take their types from their defaults; call-site
    // checks mirror evaluation's (unknown name, extra positional).
    #[test]
    fn named_parameters() {
        let f = "let f = fn(x, y = 10) => x * y in ";
        assert_clean(&format!("{}f(2)", f));
        assert_clean(&format!("{}f(2, y = 5)", f));
        assert_errors(&format!("{}f(2, z = 3)", f), &["no named parameter \"z\""]);
        assert_errors(&format!("{}f(2, 3)", f), &["extra positional parameter"]);
        // A named argument to a builtin without named parameters errors for
        // the bogus name, and — since only one of sine's two positional
        // parameters is supplied — for the missing argument too.
        assert_errors(
            "sine(440, y = 1)",
            &[
                "missing parameter of type waveform",
                "no named parameter \"y\"",
            ],
        );
    }

    #[test]
    fn missing_argument() {
        assert_errors("(fn(x, y) => x)(2)", &["missing parameter \"y\""]);
    }

    #[test]
    fn conditionals() {
        assert_errors("if 1 then 2 else 3", &["expected bool, found int"]);
        // Branches join by sort union: float and waveform join into the
        // waveform class.
        assert_clean("if true then 1 else sine(440, 0)");
    }

    #[test]
    fn constant_arithmetic_stays_integral() {
        assert_clean("nth(1 + 2, [1, 2, 3])");
    }

    #[test]
    fn seq_typing() {
        // seq(0)(w) builds a seq; `\` wants a seq on the left.
        assert_clean("seq(0)(sine(440, 0)) \\ 1");
        // A non-seq on the left of `\` is a genuine runtime error (sine
        // may fold to a constant, hence the union in the message).
        assert_errors("sine(440, 0) \\ 1", &["expected seq, found waveform"]);
        // Arithmetic threads a seq operand through (`binary_op`'s seq
        // rows), so seq-ness survives to the following `\`.
        assert_clean("(seq(0)(sine(440, 0)) * 0.5) \\ 1");
    }

    // Runtime errors the refinement lattice makes visible: arms the
    // operator tables deliberately omit, and refined contracts.
    #[test]
    fn refinement_true_positives() {
        // `binary_op` has no (Seq, Seq) arm.
        assert_errors("seq(0)(1) + seq(0)(2)", &["cannot combine two seqs with +"]);
        // `unary_op` has no Seq arm; the error pinpoints the argument
        // with the union of the domains the table does accept.
        assert_errors("-seq(0)(1)", &["expected waveform, found seq"]);
        // `reset`'s trigger accepts constants and waveforms, not seqs.
        assert_errors("reset(seq(0)(1), 1)", &["expected waveform, found seq"]);
        // `unfold`'s count is hard-checked integral at runtime.
        assert_errors("unfold(fn(x) => x, 0, 2.5)", &["expected int, found float"]);
        // `nth`'s index is hard-checked integral at runtime.
        assert_errors("nth(2.5, [1, 2, 3])", &["expected int, found float"]);
        // Contravariant contract flow: `exp` requires constants, and the
        // list supplies a definite waveform. (`sine(440, 0)` would pass
        // here: its type is the whole waveform class, because the
        // zero-frequency fold can produce a constant.)
        assert_errors("map(exp, [time])", &["expected [float], found [waveform]"]);
        // Comparisons are scalar-only at runtime.
        assert_errors("time < 1", &["expected float, found waveform"]);
        // `<>` requires seq elements; a definitely-unseq list errors.
        assert_errors("<[1, 2]>", &["expected [seq], found [int]"]);
        assert_clean("<[seq(0)(1), seq(0)(2)]>");
        // The fold of seqs is itself a seq, usable on `\`'s left — the
        // empty fold included.
        assert_clean("<[seq(0)(1), seq(0)(2)]> \\ 1");
        assert_clean("<[]> \\ 1");
    }

    // Freeman §4: definition-bound numeric functions tabulate into
    // intersections of arrows, one row per applicable atom vector.
    #[test]
    fn tabulated_definitions() {
        // Result precision at a direct call: double(2) is an int, and an
        // int is definitely not a seq.
        assert_errors(
            "let double = fn(x) => x + x in double(2) \\ 1",
            &["expected seq, found int"],
        );
        // The missing (seq, seq) arm of + is relational and survives the
        // definition boundary: double has no seq row at all.
        assert_errors(
            "let double = fn(x) => x + x in double(seq(0)(1))",
            &["expected waveform, found seq"],
        );
        // Two definite seqs get the dedicated diagnosis, same as the
        // operator itself would.
        assert_errors(
            "let add = fn(a, b) => a + b in add(seq(0)(1), seq(0)(2))",
            &["cannot combine two seqs with add"],
        );
        // Integrality flows through the definition into an int contract.
        assert_clean("let inc = fn(x) => x + 1 in nth(inc(1), [1, 2, 3])");
    }

    #[test]
    fn tabulated_argument_lambdas() {
        // In argument position the table is read distributively (the
        // union of its rows), so a mixed list stays silent...
        assert_clean("map(fn(x) => x * 0.5, [time, 2])");
        // ...and a seq element is fine where the body threads seqs...
        assert_clean("map(fn(x) => x * 2, [seq(0)(1)])");
        // ...but errors where no row accepts one: x + x has no seq row.
        // The list is checked before the table (see `check_frame`), so
        // the message shows the seq flowing into the expected arrow
        // against the rows the function actually has.
        assert_errors(
            "map(fn(x) => x + x, [seq(0)(1)])",
            &[
                "expected (seq) -> ?a, found (int) -> int ∧ (float) -> float ∧ (waveform) -> waveform",
            ],
        );
    }

    // Mixed parameter lists: numeric parameters tabulate while structural ones
    // ride along as quantified unknowns.
    #[test]
    fn tabulated_mixed_parameters() {
        // The fold's accumulator row is precise: an int seed selects the
        // int row, so the result is definitely not a seq.
        assert_errors(
            "reduce(fn(acc, x) => acc + 1, 0, [1, 2]) \\ 1",
            &["expected seq, found int"],
        );
        // The seed is checked before the table, so a float seed selects
        // the float row rather than erroring against the int row.
        assert_clean("reduce(fn(acc, x) => acc + 1, 0.5, [1, 2])");
        // A structural domain solved by the body (xs used as a list)
        // participates in selection at direct calls.
        assert_clean("let pick = fn(n, xs) => nth(n, xs) in pick(1, [1, 2])");
        assert_errors(
            "let pick = fn(n, xs) => nth(n, xs) in pick(0.5, [1, 2])",
            &["expected int, found float"],
        );
    }

    // A parameter's contract comes from the body, not the default, and named
    // numeric parameters tabulate like positional ones.
    #[test]
    fn named_parameters_type_from_the_body() {
        let f = "let f = fn(x = 100) => x + x in ";
        // An int default still admits float and waveform arguments...
        assert_clean(&format!("{}f(x = time)", f));
        assert_clean(&format!("{}f(x = 0.5)", f));
        // ...while the body's missing seq row still rejects a seq.
        assert_errors(
            &format!("{}f(x = seq(0)(1))", f),
            &["no use of f accepts (x = seq)"],
        );
        // An omitted argument takes the default; the join of the table
        // covers it, so no single row's result is claimed.
        assert_clean(&format!("{}f()", f));
        // Result precision flows through a supplied named argument.
        assert_errors(&format!("{}f(x = 2) \\ 1", f), &["expected seq, found int"]);
        // A default the body cannot accept errors at the definition.
        assert_errors("fn(x = 100) => x \\ 1", &["expected seq, found int"]);
    }

    // Beyond 4^k the two-point unseq/seq split still tabulates, keeping
    // seq relationality for wide parameter lists like the synths'.
    #[test]
    fn coarse_tabulation_beyond_the_cap() {
        let f = "let f = fn(a, b, c, d, e) => a + b + c + d + e in ";
        // A single seq threads through the fold of + rows...
        assert_clean(&format!("{}f(1, 2, 3, seq(0)(4), 5) \\ 1", f));
        // ...but two seqs have no row, even though each position alone
        // admits one.
        assert_errors(
            &format!("{}f(1, 2, 3, seq(0)(4), seq(0)(5))", f),
            &["no use of f accepts (int, int, int, seq, seq)"],
        );
    }

    // The checking policy: ground sorts judged by containment, selection
    // by atom coverage, unsolved-variable flows recorded and deferred.
    #[test]
    fn containment_judgments() {
        // A possibly-wrong value is an error: sine may fold to a float,
        // and exp requires one.
        assert_errors(
            "map(exp, [sine(440, 0)])",
            &["expected [float], found [waveform]"],
        );
        // Ground containment still passes what it should.
        assert_clean("exp(2) + sine(440, 0)");
        // Atom coverage: no single row contains waveform-or-seq, but the
        // waveform and seq atoms are covered by different rows.
        assert_clean("(if true then time else seq(0)(1)) * 1");
        // A definitely-uncovered atom still rejects.
        assert_errors("seq(0)(1) + seq(0)(2)", &["cannot combine two seqs with +"]);
        // Unconstrained parameters defer: the base pass records contracts
        // instead of judging ⊤, and the tabulated rows judge per atom.
        assert_clean("let f = fn(x) => x + 1 in f(time)");
        // Dynamic passes without imposing or cascading.
        assert_clean("debug(1) + 1");
        // A guarantee already seen is judged, deferred or not: the default
        // flows into x before the body's seq contract.
        assert_errors("fn(x = 100) => x \\ 1", &["expected seq, found int"]);
    }

    // Bound consistency is checked where guarantees and contracts meet a
    // variable — at `join_lower` and `meet_upper`, the only two mutation
    // sites — so a conflict surfaces at a flow with a span and no deferred
    // re-validation pass is needed. Bounds only tighten, making every
    // conflict final when first seen.
    #[test]
    fn bound_consistency_at_flow_sites() {
        let mut infer: Infer<()> = Infer::new();
        let Refinement::Var(id) = infer.fresh_refinement() else {
            unreachable!("fresh refinements are variables");
        };
        assert!(infer.meet_upper(id, Sort::SEQ).is_ok());
        // A guarantee escaping the contract is rejected, overlap or not,
        // as is a contract the guarantees escape.
        assert!(infer.join_lower(id, Sort::SEQ).is_ok());
        assert!(infer.join_lower(id, Sort::SEQ.union(Sort::INT)).is_err());
        assert!(infer.meet_upper(id, Sort::NON_CONST_WAVE).is_err());
    }

    // Selection against a variable that receives the arrow's own result
    // (`a → a`) iterates to a fixed point instead of committing to the seed's
    // row — the float seed's doubled results stay floats, not rejections.
    #[test]
    fn fixed_point_selection() {
        assert_clean("unfold(fn(n) => n * 2, 65.41, 9)");
        assert_clean("unfold(fn(n) => n * 2, 65.41, 9)");
        assert_clean("reduce(fn(acc, x) => acc * 0.5, 2, [1, 2])");
        // The fixed point keeps genuine sorts: a seq seed stays a seq through
        // doubling (`*` threads offsets), and a float seed's elements are
        // definitely not seqs.
        assert_clean("nth(0, unfold(fn(n) => n * 2, seq(0)(1), 9)) \\ 1");
        assert_errors(
            "nth(0, unfold(fn(n) => n * 2, 65.41, 9)) \\ 1",
            &["expected seq, found float"],
        );
    }

    #[test]
    fn unbound_variable_errors() {
        let input = "nope + 1";
        let errors = check(input);
        assert_eq!(messages(&errors), ["unbound variable 'nope'"]);
        let start = input.find("nope").unwrap();
        assert_eq!(errors[0].range(), Some(start..start + 4));

        assert_errors("nope(1) * 2", &["unbound variable 'nope'"]);
        // An unbound name recovers as Dynamic, so it produces exactly one
        // error even when applied and used in arithmetic.
        assert_errors(
            "sine(missing, wrong)",
            &["unbound variable 'missing'", "unbound variable 'wrong'"],
        );
    }

    // Unresolvable open/use bindings error at the binding; the module's
    // names then error as unbound at their use sites.
    #[test]
    fn unresolvable_modules_error() {
        let mut bindings = test_prelude();
        let (opens, errors) = parse_module::<u32, _>("open nope;\nuse also.missing;", ()).unwrap();
        assert!(errors.is_empty());
        bindings.extend(opens);
        let expr = parse_program::<u32, _>("something_from_nope + 1", ()).unwrap();
        let errors = check_program(
            |_: &[String]| Err(Error::new("no modules".to_string())),
            &bindings,
            &expr,
            None,
        );
        assert_eq!(
            messages(&errors),
            [
                "cannot load module 'nope': no modules",
                "cannot load module 'also.missing': no modules",
                "unbound variable 'something_from_nope'",
            ]
        );
    }

    #[test]
    fn program_expectations() {
        let errors = check_with_expectation("\"hello\"", Some(Expectation::Playable));
        assert_eq!(messages(&errors), ["expected numeric, found string"]);
        let errors = check_with_expectation("sine(440, 0)", Some(Expectation::Playable));
        assert!(errors.is_empty(), "got {:?}", messages(&errors));
        // A seq qualifies as a waveform program.
        let errors = check_with_expectation("seq(0)(sine(440, 0))", Some(Expectation::Playable));
        assert!(errors.is_empty(), "got {:?}", messages(&errors));
    }

    // A keys program is checked by seeding Ψ with (float, float) — the
    // note number and velocity the runtime will invoke it with — and
    // requiring a pair of waveforms back.
    #[test]
    fn note_function_expectation() {
        let errors = check_with_expectation(
            "fn(note, vel) => (sine(note*100, 0), fin(1)(sine(440, 0)))",
            Some(Expectation::NoteFunction),
        );
        assert!(errors.is_empty(), "got {:?}", messages(&errors));

        let errors = check_with_expectation(
            "fn(note) => (sine(note*100, 0), sine(note, 0))",
            Some(Expectation::NoteFunction),
        );
        assert_eq!(messages(&errors), ["extra positional parameter"]);

        let errors = check_with_expectation(
            "fn(note, vel) => (sine(note*100, 0), \"x\")",
            Some(Expectation::NoteFunction),
        );
        assert_eq!(
            messages(&errors),
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
        let errors = check_program(resolve, &bindings, &expr, None);
        assert!(errors.is_empty(), "got {:?}", messages(&errors));

        // `two` is not re-exported through `a`, so it errors as unbound.
        let expr = parse_program::<u32, _>("two", ()).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert_eq!(messages(&errors), ["unbound variable 'two'"]);
    }

    // Mirrors eval's use/projection tests: `use` binds a module type,
    // projections type-check against it, and a missing member errors.
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
        let errors = check_program(resolve, &bindings, &expr, None);
        assert!(errors.is_empty(), "got {:?}", messages(&errors));

        let expr = parse_program::<u32, _>("b.three", ()).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert_eq!(messages(&errors), ["Module has no binding 'three'"]);

        // Projecting from a non-module errors statically too.
        let expr = parse_program::<u32, _>("let x = 1 in x.y", ()).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert_eq!(
            messages(&errors),
            ["cannot project 'y' from a value of type int"]
        );
    }

    #[test]
    fn cannot_apply_non_function() {
        assert_errors("1(2)", &["cannot apply a value of type int"]);
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
        let mono = Type::function(vec![Type::float()], Type::float());
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
    // keeps only errors from its own text (dependencies re-typed along the
    // way report under their own entry). Positions come from
    // `display_with_source` so every pinned error can be read against the
    // .tuun source.
    /// The declared residue: value-level runtime errors the sort lattice
    /// cannot see (see "Toward a sound configuration" in the design doc).
    /// On a clean program, an eval error matching none of these
    /// classes is a soundness bug.
    fn declared_residue(message: &str) -> bool {
        [
            "No element with index",        // nth out of bounds
            "Invalid arguments for unfold", // negative count
            "Cannot add offsets",           // offsets not linear in time
            "Invalid arguments for filter", // empty feed-forward list
        ]
        .iter()
        .any(|class| message.contains(class))
    }

    /// A closed expression inhabiting `sort`, preferring the most general
    /// atom the sort admits, for applying exported functions.
    fn representative_text(sort: Sort) -> &'static str {
        if !sort.intersect(Sort::NON_INT_ONLY).is_empty() {
            "0.5"
        } else if !sort.intersect(Sort::INT).is_empty() {
            "2"
        } else if !sort.intersect(Sort::NON_CONST_WAVE).is_empty() {
            "time"
        } else {
            "(time | seq(time - 1))"
        }
    }

    /// The runtime sort of an evaluated value; `None` for non-numerics.
    fn runtime_sort(expr: &Expr<u32, u32>) -> Option<Sort> {
        match expr {
            Expr::Waveform(waveform::Waveform::Const(value)) => Some(if value.fract() == 0.0 {
                Sort::INT
            } else {
                Sort::NON_INT_ONLY
            }),
            Expr::Waveform(_) => Some(Sort::NON_CONST_WAVE),
            Expr::Seq { .. } => Some(Sort::SEQ),
            _ => None,
        }
    }

    /// The soundness theorem as a regression test, over the library corpus: no
    /// clean program may evaluate to an error outside the declared
    /// residue. Covers every module as a program root, and every exported
    /// function applied once per table row at representative arguments — the
    /// tabulated analog of the builtin conformance test, checking result sorts
    /// against the rows. Resolves modules for the differential harness; a
    /// generic fn item so each call site can pick its own lifetime (the
    /// synthesized binding vectors are call-site-local).
    fn diff_resolve<'a>(
        prelude: &'a [SourceBinding<u32, u32>],
        parsed: &'a [(String, Vec<SourceBinding<u32, u32>>)],
        path: &[String],
    ) -> Result<&'a [SourceBinding<u32, u32>], Error<u32>> {
        let key = path.join(".");
        if key == "__prelude" {
            return Ok(prelude);
        }
        parsed
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, bindings)| &bindings[..])
            .ok_or_else(|| Error::new(format!("no module {}", key)))
    }

    #[test]
    fn differential_library_agreement() {
        let prelude = test_prelude::<u32>();
        let mut parsed: Vec<(String, Vec<SourceBinding<u32, u32>>)> = Vec::new();
        for (index, (path, content)) in modules::EMBEDDED_MODULES.iter().enumerate() {
            let (mut bindings, errors) = parse_module::<u32, _>(content, index as u32).unwrap();
            assert!(errors.is_empty(), "parse errors in {}", path);
            bindings.insert(0, Binding::Open(vec!["__prelude".to_string()]).into());
            parsed.push((path.to_string(), bindings));
        }
        let mut calls = 0;
        let mut true_positives = 0;
        let mut residue = 0;
        // Every module as a program root: its bindings must evaluate, and
        // checking must agree.
        for (path, bindings) in &parsed {
            let expr = parse_program::<u32, _>("0", 9999).unwrap();
            let clean = check_program(
                |p: &[String]| diff_resolve(&prelude, &parsed, p),
                bindings,
                &expr,
                None,
            )
            .is_empty();
            let evaluated = eval::evaluate(
                |p: &[String]| diff_resolve(&prelude, &parsed, p),
                bindings,
                expr,
            );
            if let (true, Err(error)) = (clean, &evaluated) {
                assert!(
                    declared_residue(error.message()),
                    "soundness: module {} is clean but evaluates to: {}",
                    path,
                    error.message()
                );
            }
            calls += 1;
        }
        // Every exported function applied once per all-numeric table row.
        // The calls batch into one synthesized module per library module —
        // one check and one evaluation for all of them — with a
        // distinct source id per call for attribution.
        for (_, bindings) in &parsed {
            let mut checker: Infer<u32> = Infer::new();
            let mut context = Vec::new();
            let mut memo = HashMap::new();
            let exports = checker.build_context(
                &|p: &[String]| diff_resolve(&prelude, &parsed, p),
                bindings,
                &mut context,
                &mut memo,
            );
            // (name to bind, call text, source id, declared result sort)
            let mut applied: Vec<(String, String, u32, Option<Sort>)> = Vec::new();
            for (name, entry) in exports {
                let ContextEntry::Ty(ty) = entry else {
                    continue;
                };
                let body = match &ty {
                    Type::Forall(_, body) => &**body,
                    other => other,
                };
                let rows: Vec<&Type> = match body {
                    Type::And(rows) => rows.iter().collect(),
                    other => vec![other],
                };
                for row in rows {
                    let Type::Function {
                        positional, result, ..
                    } = row
                    else {
                        continue;
                    };
                    let domains: Option<Vec<Sort>> = positional
                        .iter()
                        .map(|domain| match domain {
                            Type::Numeric(Refinement::Ground(sort)) => Some(*sort),
                            _ => None,
                        })
                        .collect();
                    let Some(domains) = domains else {
                        continue;
                    };
                    let arguments = domains
                        .iter()
                        .map(|domain| representative_text(*domain))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let declared = match &**result {
                        Type::Numeric(Refinement::Ground(sort)) => Some(*sort),
                        _ => None,
                    };
                    let id = 10_000 + applied.len() as u32;
                    let bound = format!("__diff{}", applied.len());
                    applied.push((bound, format!("{}({})", name, arguments), id, declared));
                }
            }
            let mut extended = bindings.clone();
            for (bound, text, id, _) in &applied {
                let Ok(expr) = parse_program::<u32, _>(text, *id) else {
                    continue;
                };
                extended.push(SourceBinding::definition(
                    Pattern::Identifier(bound.clone()),
                    expr,
                ));
            }
            calls += applied.len();
            let expr = parse_program::<u32, _>("0", 9999).unwrap();
            let errors = check_program(
                |p: &[String]| diff_resolve(&prelude, &parsed, p),
                &extended,
                &expr,
                None,
            );
            let flagged: Vec<u32> = errors.iter().filter_map(|error| error.source()).collect();
            // Classifies one call by evaluating it alone against the
            // module. An eval error's span points into the failing body,
            // not the call, so per-call evaluation is the only reliable
            // attribution; it runs for flagged calls and as the fallback
            // when the batch fails.
            let mut classify = |text: &str, id: u32, declared: Option<Sort>| {
                let expr = parse_program::<u32, _>(text, id).unwrap();
                let evaluated = eval::evaluate(
                    |p: &[String]| diff_resolve(&prelude, &parsed, p),
                    bindings,
                    expr,
                );
                match &evaluated {
                    Err(error) if declared_residue(error.message()) => residue += 1,
                    Err(error) => {
                        assert!(
                            flagged.contains(&id),
                            "soundness: {:?} is clean but evaluates to: {}",
                            text,
                            error.message()
                        );
                        true_positives += 1;
                    }
                    Ok(value) => {
                        if let (Some(declared), Some(actual)) =
                            (declared, runtime_sort(&value.expr))
                        {
                            assert!(
                                actual.is_subset(declared),
                                "{:?}: runtime sort {} outside the row's declared {}",
                                text,
                                actual,
                                declared
                            );
                        }
                    }
                }
            };
            // Flagged calls are expected to fail: evaluate each alone.
            for (_, text, id, declared) in &applied {
                if flagged.contains(id) {
                    classify(text, *id, *declared);
                }
            }
            // The unflagged calls batch into one evaluation; a batch
            // failure falls back to per-call attribution.
            let mut batch = bindings.clone();
            for (bound, text, id, _) in &applied {
                if !flagged.contains(id)
                    && let Ok(expr) = parse_program::<u32, _>(text, *id)
                {
                    batch.push(SourceBinding::definition(
                        Pattern::Identifier(bound.clone()),
                        expr,
                    ));
                }
            }
            match eval::evaluate_bindings(|p: &[String]| diff_resolve(&prelude, &parsed, p), &batch)
            {
                Ok(evaluated) => {
                    for (bound, text, id, declared) in &applied {
                        if flagged.contains(id) {
                            continue;
                        }
                        let Some((_, value)) =
                            evaluated.iter().rev().find(|(name, _)| name == bound)
                        else {
                            continue;
                        };
                        if let (Some(declared), Some(actual)) =
                            (declared, runtime_sort(&value.expr))
                        {
                            assert!(
                                actual.is_subset(*declared),
                                "{:?}: runtime sort {} outside the row's declared {}",
                                text,
                                actual,
                                declared
                            );
                        }
                    }
                }
                Err(_) => {
                    for (_, text, id, declared) in &applied {
                        if !flagged.contains(id) {
                            classify(text, *id, *declared);
                        }
                    }
                }
            }
        }
        println!(
            "differential: {} programs, {} true positives, {} residue hits",
            calls, true_positives, residue
        );
        // The harness must be exercising real applied rows.
        assert!(calls >= 40, "only {} programs checked", calls);
    }

    /// Prints the checker's findings over the embedded library (run with
    /// `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn library_report() {
        let prelude = test_prelude::<u32>();
        let mut parsed: Vec<(String, &str, Vec<SourceBinding<u32, u32>>)> = Vec::new();
        for (index, (path, content)) in modules::EMBEDDED_MODULES.iter().enumerate() {
            let (mut bindings, _) = parse_module::<u32, _>(content, index as u32).unwrap();
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
        for (index, (path, content, bindings)) in parsed.iter().enumerate() {
            let errors = check_program(&resolve, bindings, &expr, None);
            let own: Vec<String> = errors
                .iter()
                .filter(|error| error.source() == Some(index as u32))
                .map(|error| error.display_with_source(content))
                .collect();
            println!("=== {} ({} errors)", path, own.len());
            for error in own {
                println!("  {}", error);
            }
        }
    }

    #[test]
    fn embedded_modules_pinned_errors() {
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
            let errors = check_program(&resolve, bindings, &expr, None);
            for error in &errors {
                if error.source() == Some(index as u32) {
                    report.push(format!("{}: {}", path, error.display_with_source(content)));
                }
            }
        }
        // The refinement lattice checks the entire embedded library without a
        // single error. Anything appearing here needs source-level
        // verification before being accepted as a new pin.
        assert_eq!(report, Vec::<String>::new());
    }
}
