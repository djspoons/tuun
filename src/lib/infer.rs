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
//! - A projection is resolved against the module's own bindings, so it
//!   requires the checker to know *which* module: an expression whose type
//!   is still unknown is rejected rather than trusted. The type is tracked
//!   wherever it flows — a `let`, a directly applied lambda, a list or tuple
//!   element — so what this rules out is a module arriving as an unannotated
//!   parameter, and with it functions over modules. In exchange, every
//!   projection that survives checking is one whose name was looked up, and
//!   a misspelled one is a static error wherever it appears.
//!
//! TODO add notes about default values and how they interact with refinements

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

/// Returns the type of the identifier covering `offset` in `expr`, rendered
/// as diagnostics render types, or `None` when no identifier covers it.
///
/// Checks under the same bindings and expectation as [`check_program`], so
/// the answer is the one the checker itself is working with, and reports it
/// only once inference has finished: a parameter whose type the body settles
/// reads as what it was solved to rather than as an unknown. Errors elsewhere
/// in the program do not prevent an answer — inference recovers and carries
/// on — and the errors themselves are discarded, since `check_program` is
/// what reports them.
///
/// `offset` is a byte offset into the text `expr` was parsed from. Only
/// identifiers answer: a name's type is what the context holds for it,
/// which is well defined in a way an arbitrary sub-expression's is not (a
/// function's node under an application judges to the call's result).
pub fn type_at<'a, M, S, F>(
    resolve: F,
    bindings: &'a [SourceBinding<M, S>],
    expr: &SourceExpr<M, S>,
    expectation: Option<Expectation>,
    offset: usize,
) -> Option<String>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    S: Clone,
{
    let mut checker = Infer::new();
    let mut context = Vec::new();
    let mut memo = HashMap::new();
    // The probe is armed only after the context is built: the modules typed
    // there carry offsets into their own sources, which the probe cannot
    // tell from this one's.
    checker.build_context(&resolve, bindings, &mut context, &mut memo);
    checker.probe = Some(offset);
    let mut psi = match expectation {
        Some(Expectation::NoteFunction) => vec![Frame {
            positional: vec![
                (Type::int(), expr.span.clone()),
                (Type::float(), expr.span.clone()),
            ],
            named: Vec::new(),
            span: expr.span.clone(),
        }],
        _ => Vec::new(),
    };
    checker.infer(&mut context, &mut psi, expr);
    let (_, ty) = checker.probed.take()?;
    Some(checker.display(&ty))
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
const MAX_TABULATION_VECTORS: usize = 512;

/// The most rounds selection may take to reach a fixed point.
///
/// Each round that changes anything grows a variable's guarantees within the
/// four-atom lattice, so convergence is bounded by its height; this is the
/// backstop, not the budget.
const MAX_SELECTION_ROUNDS: usize = 8;

/// The four atoms of the sort lattice, most specific first.
const FULL_ATOMS: [Sort; 4] = [
    Sort::INT,
    Sort::NON_INT_ONLY,
    Sort::NON_CONST_WAVE,
    Sort::SEQ,
];

/// The two-point unseq/seq split: the soundness-critical distinction, kept
/// when the full basis does not fit the budget.
const COARSE_ATOMS: [Sort; 2] = [Sort::WAVE, Sort::SEQ];

/// How selection decided one intersection against one frame.
enum Selection {
    /// A conjunct applied; the application's result type.
    Selected(Type),
    /// No conjunct has the frame's shape; selection does not apply, and callers
    /// fall back to plain conjunct subtyping.
    NoMatchingConjuncts,
    /// Conjuncts match the frame's shape but none accepts the arguments — the
    /// runtime has no matching arm.
    NoApplicableConjunct,
}

/// The algorithmic state threaded through every judgment: Xie and
/// Oliveira's substitution `S` and name supply `N` (Appendix E.1), plus the
/// errors accumulated so far.
struct Infer<S> {
    /// The last sort conflict a refinement variable reported, as
    /// (guaranteed, required).
    ///
    /// A breadcrumb for diagnostics only, never rolled back: it is cleared
    /// at the start of each [`Infer::subtype_check`] and read only when that
    /// check fails, so what it holds is a conflict from the failing check.
    conflict: Option<(Sort, Sort)>,
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
    /// The offset a [`type_at`] query is asking about, once inference has
    /// reached the text the offset indexes into. `None` for a plain check,
    /// and while building the context — the modules typed there carry
    /// offsets into their own sources, which this cannot tell apart.
    probe: Option<usize>,
    /// The narrowest identifier found covering `probe`, with the width of
    /// its span. Held unrendered: a parameter's type is a meta until the
    /// body settles it, so only the finished substitution reads right.
    probed: Option<(usize, Type)>,
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

/// Identifies a module by the slice it was resolved to.
///
/// Both halves matter: a start address alone would confuse two modules that
/// share one — every empty module has the same dangling pointer, and a slice
/// may begin where a longer one does.
type ModuleKey = (usize, usize);

/// What a module exports, once built; `None` while it is still being built,
/// which is how a cycle is caught.
type ModuleMemo = HashMap<ModuleKey, Option<Vec<(String, ContextEntry)>>>;

/// Returns the key identifying `module`.
fn module_key<M, S>(module: &[SourceBinding<M, S>]) -> ModuleKey {
    (module.as_ptr() as usize, module.len())
}

impl<S: Clone> Infer<S> {
    fn new() -> Infer<S> {
        Infer {
            subst: HashMap::new(),
            supply: 0,
            refs: Vec::new(),
            journal: Vec::new(),
            errors: Vec::new(),
            conflict: None,
            probe: None,
            probed: None,
        }
    }

    fn error(&mut self, message: String, span: &Option<Span<S>>) {
        self.errors.push(Error::types(message, span.clone()));
    }

    /// Records `ty` as the answer to a pending type query when `span` covers
    /// the offset the query asks about.
    ///
    /// The narrowest span wins, so the name in `synth.piano` answers for
    /// itself rather than for the projection that encloses it.
    fn probe_type(&mut self, span: &Option<Span<S>>, ty: &Type) {
        let (Some(offset), Some(span)) = (self.probe, span) else {
            return;
        };
        if !span.range.contains(&offset) {
            return;
        }
        let width = span.range.len();
        let narrower = match &self.probed {
            Some((best, _)) => width <= *best,
            None => true,
        };
        if narrower {
            self.probed = Some((width, ty.clone()));
        }
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
                let lower = self.refs[*id as usize].lower;
                if !lower.is_subset(contract) {
                    self.conflict = Some((lower, contract));
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
    ///
    /// Two ground sorts have no bounds to merge, so they are judged
    /// directly, and an invariant position judges them by equality: `int`
    /// and `seq` are different types there, not merely overlapping ones.
    // TODO two unsolved variables are not linked: guarantees or contracts
    // arriving at one after this merge do not reach the other. A union-find
    // over refinement variables would close this — MLsub's biunification
    // handles a variable-variable constraint exactly so (merge the bounds,
    // alias the variables).
    fn numeric_unify(&mut self, x: &Refinement, y: &Refinement) -> Result<(), ()> {
        if let (Refinement::Ground(x), Refinement::Ground(y)) = (x, y) {
            return if x == y { Ok(()) } else { Err(()) };
        }
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

    /// Returns `ty` with the refinement variables at or above `floor`
    /// replaced by the sorts they may currently inhabit.
    ///
    /// A `floor` of 0 resolves every variable, which is what rendering
    /// wants. A higher floor grounds only the variables allocated since
    /// that point, leaving older ones live: a type carried out of a
    /// speculative attempt must not mention variables the attempt's
    /// rollback is about to pop, but variables that predate the attempt
    /// outlive it and stay linked.
    fn resolve_refinements(&self, ty: &Type, floor: u32) -> Type {
        match ty {
            Type::Numeric(rep @ Refinement::Var(id)) if *id >= floor => {
                Type::ground(self.may_of(rep))
            }
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional
                    .iter()
                    .map(|t| self.resolve_refinements(t, floor))
                    .collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), self.resolve_refinements(t, floor)))
                    .collect(),
                result: Box::new(self.resolve_refinements(result, floor)),
            },
            Type::And(conjuncts) => Type::And(
                conjuncts
                    .iter()
                    .map(|t| self.resolve_refinements(t, floor))
                    .collect(),
            ),
            Type::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|t| self.resolve_refinements(t, floor))
                    .collect(),
            ),
            Type::List(item) => Type::List(Box::new(self.resolve_refinements(item, floor))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), self.resolve_refinements(t, floor)))
                    .collect(),
            ),
            Type::Forall(vars, body) => Type::Forall(
                vars.clone(),
                Box::new(self.resolve_refinements(body, floor)),
            ),
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
        self.resolve_refinements(&ty.apply(&self.subst), 0)
            .to_string()
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
    ///
    /// The base pass is exploratory whenever tabulation succeeds, and its
    /// errors give way to the conjuncts'. Its parameters are unsolved, so it has
    /// to summarize every atom they admit at once — `2 * freq` is "some
    /// numeric" there — and a summary no single call will ever see must not
    /// be the thing that reports. What the conjuncts keep instead is the per-atom
    /// verdict: a vector whose body errors contributes no conjunct, and a body
    /// that errors at *every* vector leaves no conjuncts at all, so an
    /// unconditional error still surfaces through the base type.
    fn infer_definition<M>(&mut self, context: &mut TypeContext, expr: &SourceExpr<M, S>) -> Type {
        let start = self.errors.len();
        let base = self.infer(context, &mut Vec::new(), expr);
        let explored = self.errors.len();
        match self.tabulate(context, expr, &base, None) {
            Some(table) => {
                // Drop what the base pass said, keeping what tabulation
                // said: the conjuncts re-check the body per atom, and the
                // defaults — which depend on no parameter — are judged once
                // inside `tabulate`, so anything unconditional is reported
                // there rather than here.
                self.errors.drain(start..explored);
                table
            }
            None => base,
        }
    }

    /// The atom basis a whole curried spine fits in, or `None` when even the
    /// coarse split does not fit and the definition is left untabulated.
    fn spine_basis<M>(&self, expr: &SourceExpr<M, S>, base: &Type) -> Option<&'static [Sort]> {
        [&FULL_ATOMS[..], &COARSE_ATOMS[..]]
            .into_iter()
            .find(|atoms| {
                self.spine_vectors(expr, base, atoms)
                    .is_some_and(|vectors| vectors <= MAX_TABULATION_VECTORS)
            })
    }

    /// The vectors a curried spine enumerates with `atoms` as its basis: the
    /// product across every lambda in it, since each inner one is tabulated
    /// again for every vector of the lambda outside it.
    fn spine_vectors<M>(
        &self,
        expr: &SourceExpr<M, S>,
        ty: &Type,
        atoms: &[Sort],
    ) -> Option<usize> {
        let Expr::Function {
            positional,
            named,
            body,
        } = &expr.expr
        else {
            return Some(1);
        };
        let Type::Function {
            positional: parameters,
            named: named_parameters,
            result,
        } = self.resolve(ty)
        else {
            return Some(1);
        };
        if parameters.len() != positional.len() || named_parameters.len() != named.len() {
            return Some(1);
        }
        let numeric: Vec<bool> = parameters
            .iter()
            .map(|parameter| matches!(self.resolve(parameter), Type::Numeric(_)))
            .collect();
        let named_numeric: Vec<bool> = named_parameters
            .iter()
            .map(|(_, parameter)| matches!(self.resolve(parameter), Type::Numeric(_)))
            .collect();
        Self::level_vectors(atoms, &numeric, &named_numeric)?
            .checked_mul(self.spine_vectors(body, &result, atoms)?)
    }

    /// The vectors one lambda enumerates: one choice per numeric positional
    /// parameter, and one more than that per named parameter, which may also
    /// be omitted.
    fn level_vectors(atoms: &[Sort], numeric: &[bool], named_numeric: &[bool]) -> Option<usize> {
        let positional = numeric.iter().filter(|numeric| **numeric).count();
        let mut vectors = atoms.len().checked_pow(u32::try_from(positional).ok()?)?;
        for numeric in named_numeric {
            let radix = if *numeric { atoms.len() + 1 } else { 2 };
            vectors = vectors.checked_mul(radix)?;
        }
        Some(vectors)
    }

    /// Tabulates a definition-bound function over its numeric parameters —
    /// Freeman and Pfenning §4: the principal refinement type of a definition
    /// is a finite intersection of arrows, found by re-checking the body at
    /// each point of the finite refinement lattice, here each vector of atoms
    /// over the numeric parameters (positional and named alike). Non-numeric
    /// parameters ride along as fresh unknowns (solved per conjunct by the
    /// body, quantified by the caller's generalization), and a non-numeric
    /// named parameter's default flows in as a guarantee, as in the base pass.
    /// A vector whose body check errors contributes no conjunct (the function
    /// is not applicable there), and the exploratory errors are discarded — the
    /// base pass has already reported anything unconditional.
    ///
    /// Beyond `MAX_TABULATION_VECTORS`, enumeration retries on the two-point
    /// unseq/seq split — keeping the relational seq holes, the
    /// soundness-critical part, at the cost of the int/float distinctions — and
    /// returns `None` (keep the base type, the freeze-at-generalize summary)
    /// only past that, or for functions with no numeric parameters.
    ///
    /// This is what makes parameter contracts *relational* rather than
    /// per-position: `fn(a, b) => a + b` gets no `(seq, seq)` conjunct, so a
    /// two-seq call errors even though each position separately admits a seq.
    fn tabulate<M>(
        &mut self,
        context: &mut TypeContext,
        expr: &SourceExpr<M, S>,
        base: &Type,
        basis: Option<&'static [Sort]>,
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
        // Every named parameter is worth a table even when nothing is numeric,
        // because a call may always omit one: the omitted conjunct is where the
        // body is checked at the default's own type, and the supplied conjunct
        // is where it is checked at whatever the caller brings.
        let tabulated = numeric
            .iter()
            .chain(&named_numeric)
            .filter(|numeric| **numeric)
            .count();
        if tabulated == 0 && named.is_empty() {
            return None;
        }
        // The atom basis is chosen once for a whole curried spine and passed
        // down, because an inner lambda is re-tabulated once per outer
        // vector: the spine costs the *product* of its levels. Choosing per
        // lambda lets a curried definition evade the budget entirely, since
        // no level need exceed it while their product does — `fn(a, b, c) =>
        // fn(d, e) => ...` is 4³ × 4² = 1024 vectors, four times what the
        // same parameters cost written flat.
        let atoms = match basis {
            Some(atoms) => atoms,
            None => self.spine_basis(expr, base)?,
        };
        let vectors = Self::level_vectors(atoms, &numeric, &named_numeric)?;
        // Refinement variables reachable from the enclosing context belong to
        // outer scopes and must stay live in the conjuncts (mirroring
        // `generalize`'s exclusion).
        let mut keep = Vec::new();
        for (_, entry) in context.iter() {
            if let ContextEntry::Ty(ty) = entry {
                ty.free_refinements(&self.subst, &mut keep);
            }
        }
        // A type query is not answered from here. Each vector re-checks the
        // body with the parameters pinned to one hypothetical atom, so an
        // identifier would report whichever conjunct happened to run last, and
        // the conjunct's refinement variables are popped by its rollback
        // anyway.
        let probe = self.probe.take();
        // Defaults are inferred once, here: they are evaluated in the enclosing
        // scope, so they see neither the parameters nor the conjunct's atoms,
        // and an error in one is unconditional rather than something a vector
        // gets to re-decide. Inferring them inside the loop would also let a
        // default see a positional parameter, which neither `infer` nor
        // evaluation allows.
        let default_types: Vec<Type> = named
            .iter()
            .map(|(_, default)| self.infer(context, &mut Vec::new(), default))
            .collect();
        let mut conjuncts = Vec::new();
        for index in 0..vectors {
            let errors = self.errors.len();
            let mark = self.mark();
            let depth = context.len();
            let mut stride = 1usize;
            let mut domains = Vec::with_capacity(positional.len());
            for (pattern, numeric) in positional.iter().zip(&numeric) {
                if *numeric {
                    let atom = atoms[(index / stride) % atoms.len()];
                    stride *= atoms.len();
                    self.bind_pattern(context, pattern, Type::ground(atom), &expr.span, false);
                    domains.push(Type::ground(atom));
                } else {
                    domains.push(self.pattern_param_type(context, pattern));
                }
            }
            let mut named_domains = Vec::with_capacity(named.len());
            for (((name, _), numeric), default_ty) in
                named.iter().zip(&named_numeric).zip(&default_types)
            {
                let radix = if *numeric { atoms.len() + 1 } else { 2 };
                let choice = (index / stride) % radix;
                stride *= radix;
                // The last choice is the omitted one: the parameter takes the
                // value a call that leaves it out would get, so the body is
                // checked at the default's own type. Leaving the name off the
                // conjunct is what tells selection the conjunct is for such a
                // call — see `named_matches`.
                let omitted = choice == radix - 1;
                let parameter = if omitted {
                    default_ty.clone()
                } else if *numeric {
                    Type::ground(atoms[choice])
                } else {
                    // The default does not constrain a supplied argument: this
                    // conjunct is the call that brings its own, and the omitted
                    // conjunct above is where the default is judged.
                    self.fresh_meta()
                };
                context.push((name.clone(), ContextEntry::Ty(parameter.clone())));
                if !omitted {
                    named_domains.push((name.clone(), parameter));
                }
            }
            // A lambda body is itself tabulated: Freeman and Pfenning's ABS
            // applies at every abstraction, not once per definition, so a
            // curried function gets an intersection at each arrow rather than
            // only at the outermost. Its base pass summarizes over parameters
            // this conjunct leaves unsolved, so those errors are exploratory
            // too and must not decide whether this conjunct stands.
            let inner = self.errors.len();
            let result = self.infer(context, &mut Vec::new(), body);
            let explored = self.errors.len();
            let result = match self.tabulate(context, body, &result, Some(atoms)) {
                Some(table) => {
                    self.errors.drain(inner..explored);
                    table
                }
                None => result,
            };
            context.truncate(depth);
            if self.errors.len() == errors {
                // Resolve the conjunct fully before rolling back the state it
                // was solved in.
                let conjunct = Type::Function {
                    positional: domains,
                    named: named_domains,
                    result: Box::new(result),
                }
                .apply(&self.subst);
                conjuncts.push(self.freeze_refinements(&conjunct, true, &keep));
            }
            self.errors.truncate(errors);
            self.rollback(mark);
        }
        self.probe = probe;
        if conjuncts.is_empty() {
            return None;
        }
        // A call may always leave a named argument out, so the vectors that
        // omit one are not optional the way an atom's are: an atom a conjunct
        // cannot take is a call the caller can avoid making, but omitting is
        // always available. If no conjunct stands with the name left off, the
        // default does not work and every such call is unanswerable — report it
        // here rather than hand back a table that promises a parameter it
        // cannot honour.
        for (name, default) in named {
            let omitted_stands = conjuncts.iter().any(|conjunct| match conjunct {
                Type::Function { named, .. } => !named.iter().any(|(other, _)| other == name),
                _ => false,
            });
            if !omitted_stands {
                let message = format!("default value for \"{}\" cannot be used in the body", name);
                self.error(message, &default.span);
            }
        }
        let mut conjuncts = merge_conjuncts(conjuncts);
        Some(if conjuncts.len() == 1 {
            conjuncts.remove(0)
        } else {
            Type::And(conjuncts)
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
            // Two tables unify conjunct by conjunct. Without this arm an
            // intersection unified with *itself* failed, since nothing below
            // matches a pair of them.
            (Type::And(xs), Type::And(ys)) => {
                if xs.len() != ys.len() {
                    return Err(());
                }
                for (x, y) in xs.clone().iter().zip(ys.clone()) {
                    self.unify(x, &y)?;
                }
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
            // cannot poison a sibling argument: when no single conjunct applies
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
                // Selection needs something to select *on*. Xue, Oliveira
                // and Xie's applicative subtyping matches a conjunct against
                // "just the function argument type or a label, instead of the
                // complete type"; where no parameter carries a sort there is
                // no such selector, and coverage degenerates to the join of
                // every conjunct — honest, and saying nothing. Sub-And-L instead
                // checks the result, which by then may be constrained.
                //
                // Removing this leaves the embedded library checking clean,
                // so it is no longer load-bearing there — C2, C4 and C6
                // closed those errors by other means. What it still buys is
                // the diagnostic: without it the shape in
                // `a_chord_rejects_an_instrument_it_cannot_mix` reports a
                // spurious `expected [waveform], found [numeric]` from the
                // degenerate join, on top of the real error.
                let selectable = positional
                    .iter()
                    .any(|parameter| match self.resolve(parameter) {
                        Type::Numeric(rep) => !self.definite_of(&rep).is_empty(),
                        Type::Meta(_) => false,
                        _ => true,
                    });
                if !selectable {
                    return self.subtype_any_conjunct(conjuncts.clone(), &b);
                }
                // Selecting against unsolved variables iterates to a fixed
                // point (Freeman's abstract interpretation): when the arrow's
                // result flows back into a parameter variable — `unfold`'s `a →
                // a`, `reduce`'s accumulator — the growth can change which conjuncts
                // apply and what contracts they may impose, so a one-shot
                // selection over-commits (e.g. a float seed selects the float
                // conjunct, whose result grows the variable and then violates the
                // conjunct's own contract). Roll the attempt back, carry the growth,
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
                        Selection::Selected(conjunct_result) => {
                            self.subtype(&conjunct_result, result)
                        }
                        Selection::NoApplicableConjunct => Err(()),
                        Selection::NoMatchingConjuncts => {
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
                    if before == after {
                        break outcome;
                    }
                    // Each round that does not settle grows some variable's
                    // guarantees, and a sort has four atoms, so one variable
                    // can grow at most four times before it is ⊤. The bound
                    // is a backstop against a mistake in that reasoning, not
                    // a budget: reaching it means the iteration is not the
                    // ascending chain it is meant to be, and the result
                    // would be whatever the last round happened to leave.
                    debug_assert!(
                        iterations < MAX_SELECTION_ROUNDS,
                        "selection did not converge within {} rounds",
                        MAX_SELECTION_ROUNDS
                    );
                    if iterations >= MAX_SELECTION_ROUNDS {
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
            // A bare meta gives selection nothing to select on, so the
            // intersection binds whole. Neither Sub-And rule fits: nothing
            // here says which conjunct is used, and requiring the meta to
            // meet every conjunct would solve it against the first and then
            // fail against the rest.
            (Type::And(_), Type::Meta(_)) | (Type::Meta(_), Type::And(_)) => self.unify(&a, &b),
            // Meeting an intersection requires meeting every conjunct
            // (rule Sub-And-R). It is tried before Sub-And-L below because
            // it is invertible and Sub-And-L is not: an intersection meeting
            // an intersection may need a *different* left conjunct for each
            // right one, which committing to one conjunct first forecloses.
            (_, Type::And(conjuncts)) => {
                for conjunct in conjuncts.clone() {
                    self.subtype(&a, &conjunct)?;
                }
                Ok(())
            }
            // An intersection is usable where any of its conjuncts is
            // (rule Sub-And-L); failed attempts roll back their partial
            // solving.
            //
            // Expected arrows *with named parameters* deliberately land
            // here too (the guard above): a pseudo-frame models a call,
            // and a promised named parameter is not a call argument — as
            // an "omitted argument" it would skip its contravariant check,
            // and a conjunct lacking the name would wrongly stay eligible. The
            // per-conjunct path does the full named discipline per conjunct,
            // with the first fitting conjunct committing — no distributive
            // reading, no fixed-point iteration.
            // TODO extend the pseudo-frame with promised named parameters
            // (and filter conjuncts to those offering them) so named-having
            // arrows join the selection path.
            (Type::And(conjuncts), _) => self.subtype_any_conjunct(conjuncts.clone(), &b),
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
            // A structure meeting a meta solves it to a same-shaped
            // skeleton of fresh metas and continues into that, so every
            // leaf reaches an arm that preserves the direction. Falling
            // through to `unify` would solve the meta to the structure
            // itself, and a leaf frozen at a ground sort can no longer
            // widen the way one met on its own does — nor, for an arrow,
            // vary the way the arm below lets its parameters and result
            // vary. This is Xie and Oliveira's arrow unification (Fig. 14,
            // rule AF-Mono), which their AS-FunR/AS-FunL call for exactly
            // this, and their AS-PairL/AS-PairR extend to structures.
            (Type::Tuple(_) | Type::List(_) | Type::Function { .. }, Type::Meta(id))
                if !self.occurs(*id, &a) =>
            {
                let skeleton = self.skeleton(&a);
                self.solve(*id, skeleton.clone());
                self.subtype(&a, &skeleton)
            }
            (Type::Meta(id), Type::Tuple(_) | Type::List(_) | Type::Function { .. })
                if !self.occurs(*id, &b) =>
            {
                let skeleton = self.skeleton(&b);
                self.solve(*id, skeleton.clone());
                self.subtype(&skeleton, &b)
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
            // A meta anywhere else has no subtyping left to do, only a
            // solution to record, so unification takes it: AU-Refl for two
            // of the same, AU-Var1/AU-Var2 otherwise, under the occurs
            // check. The pairings where the direction *does* matter —
            // against a numeric, an intersection, or a structure — are the
            // arms above, and a structure whose occurs check failed there
            // arrives here to fail on the same condition.
            (Type::Meta(_), _) | (_, Type::Meta(_)) => self.unify(&a, &b),
            // AS-Mono, the monotypes: a base type is a subtype of itself,
            // and a rigid variable of no other variable.
            (Type::Bool, Type::Bool) | (Type::String, Type::String) => Ok(()),
            (Type::Var(x), Type::Var(y)) if x == y => Ok(()),
            // Modules are invariant. `use` is the only way to bind one, so
            // two module types meet only where they are the same module,
            // exported binding for exported binding.
            (Type::Module(_), Type::Module(_)) => self.unify(&a, &b),
            // Nothing relates the rest: two different constructors, or two
            // rigid variables that are not the same one. `Dynamic`,
            // `Forall`, and `And` never reach here — the arms above take
            // them on either side.
            _ => Err(()),
        }
    }

    /// Whether the meta `id` appears free in `ty` — AU-Var1's `α̂ ∉ ftv(S τ)`.
    fn occurs(&self, id: u32, ty: &Type) -> bool {
        let mut metas = Vec::new();
        ty.free_metas(&self.subst, &mut metas);
        metas.contains(&id)
    }

    /// Returns a type of the same shape as `ty` whose leaves are fresh metas.
    ///
    /// Structure is copied; everything under it is left to be solved.
    ///
    /// # Example
    ///
    /// The skeleton of `(int, [waveform])` is `(?a, [?b])`.
    fn skeleton(&mut self, ty: &Type) -> Type {
        match self.resolve(ty) {
            Type::Tuple(items) => {
                let mut skeletons = Vec::with_capacity(items.len());
                for item in &items {
                    skeletons.push(self.skeleton(item));
                }
                Type::Tuple(skeletons)
            }
            Type::List(item) => Type::List(Box::new(self.skeleton(&item))),
            Type::Function {
                positional,
                named,
                result,
            } => {
                let mut parameters = Vec::with_capacity(positional.len());
                for parameter in &positional {
                    parameters.push(self.skeleton(parameter));
                }
                let mut named_parameters = Vec::with_capacity(named.len());
                for (name, parameter) in &named {
                    let parameter = self.skeleton(parameter);
                    named_parameters.push((name.clone(), parameter));
                }
                Type::Function {
                    positional: parameters,
                    named: named_parameters,
                    result: Box::new(self.skeleton(&result)),
                }
            }
            _ => Type::Meta(self.fresh_id()),
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

    /// Returns the type a named parameter starts from, with `default_ty` flowed
    /// into it.
    ///
    /// A numeric default flows into an unknown sort as a guarantee, leaving the
    /// parameter free to widen to whatever the body needs. A non-numeric
    /// default becomes the parameter's type outright. The body is checked once,
    /// so a numeric leaf left open could be widened by a later call after the
    /// result computed from it was already fixed.
    ///
    /// # Example
    ///
    /// `fn(k = 2) => k` starts `k` at an unknown guaranteeing `int`, while
    /// `fn(k = [1, 2]) => k` starts it at `[int]`.
    fn default_parameter(&mut self, default_ty: &Type, span: &Option<Span<S>>) -> Type {
        // Through the quantifier first: a default bound by a definition is
        // generalized, and the parameter takes an instance of it rather
        // than the quantified type, so the shape to pin is the instance's.
        let resolved = match self.resolve(default_ty) {
            Type::Forall(vars, body) => self.instantiate(&vars, *body),
            resolved => resolved,
        };
        let parameter = match resolved {
            compound @ (Type::Tuple(_) | Type::List(_) | Type::Function { .. }) => compound,
            _ => self.fresh_meta(),
        };
        self.subtype_check(default_ty, &parameter, span);
        parameter
    }

    /// Checks `found <: expected` and reports an error at `span` on
    /// failure, rolling back any partial solving from the failed attempt.
    fn subtype_check(&mut self, found: &Type, expected: &Type, span: &Option<Span<S>>) {
        let mark = self.mark();
        self.conflict = None;
        if self.subtype(found, expected).is_err() {
            self.rollback(mark);
            let message = match self.nested_conflict(found, expected) {
                Some((guaranteed, required)) => format!(
                    "no sort fits here: guaranteed {}, required {}",
                    guaranteed, required
                ),
                None => format!(
                    "expected {}, found {}",
                    self.display(expected),
                    self.display(found)
                ),
            };
            self.error(message, span);
        }
    }

    /// Returns the sort conflict to report in place of the two types, when
    /// naming them would not show where they disagree.
    ///
    /// A conflict on a numeric *nested* inside a structure cannot be seen in
    /// an `expected ..., found ...` message: both sides render whole, and the
    /// offending leaf is one position among several — often rendering as a
    /// plausible sort, since a variable displays the guarantees it has
    /// collected rather than the conflict it is in. Where either side *is* a
    /// numeric, that side's name states its sort outright and the ordinary
    /// message says more than this one.
    ///
    /// # Example
    ///
    /// A leaf conflict inside an arrow reports `no sort fits here: guaranteed
    /// numeric, required waveform` rather than naming two arrows that read as
    /// though they should match.
    fn nested_conflict(&self, found: &Type, expected: &Type) -> Option<(Sort, Sort)> {
        let conflict = self.conflict?;
        let bare = |ty: &Type| matches!(self.resolve(ty), Type::Numeric(_));
        (!bare(found) && !bare(expected)).then_some(conflict)
    }

    /// Returns the join of the two types: unification if they agree
    /// structurally, sort union for numerics (mixed float/waveform/seq lists
    /// and branches are common), pointwise for lists and tuples, with the
    /// variance for arrows, pairwise for two intersections, and `Dynamic`
    /// with an error where there is none. Quantified types join at an
    /// instance.
    ///
    /// Xie and Oliveira have no join, having no subtyping between base
    /// types, but the refinement line does. Freeman and Pfenning give unions
    /// a type constructor of their own — "such union types arise, for
    /// example, from the different branches of an `if` expression" — in a
    /// union normal form `unf ::= inf | unf ∨ unf` where an arrow reads
    /// `inf → unf`: an intersection accepted, a union returned. Sorts are a
    /// union already, so that shape falls out here rather than needing a
    /// constructor, and the join of two tables produces it directly
    /// (`(int) -> int or waveform`). What tuun has no representation for is a
    /// union of two *structures*, which is why a pair with nothing in common
    /// is an error rather than a type.
    ///
    /// The law behind the intersection case is Davies' (§4.5):
    ///
    /// ```text
    /// (R1 → S1) & (R2 → S2)  ≤  (R1 ∨ R2) → (S1 ∨ S2)
    /// ```
    ///
    /// He applied it "to each pair of conjuncts", as the intersection case
    /// below does, and withdrew it: his lattice has no union operation, so
    /// `R1 ∨ R2` had to be built by enumerating the common upper bounds,
    /// "infeasible for even simple higher-order sorts". Here a union of sorts
    /// is one instruction over four atoms, so the same algorithm is cheap.
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
            // Arrows join pointwise with the variance: a function usable as
            // either accepts only what both accept and returns what either
            // returns. A named parameter survives only when both offer it,
            // since the joined type promises it to every caller. Where a
            // parameter has no meet the two share no arrow at all, and the
            // fallthrough reports them incompatible.
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
            ) if p1.len() == p2.len() => {
                let (p1, n1, r1) = (p1.clone(), n1.clone(), *r1.clone());
                let (p2, n2, r2) = (p2.clone(), n2.clone(), *r2.clone());
                let mut positional = Vec::with_capacity(p1.len());
                for (x, y) in p1.iter().zip(&p2) {
                    match self.meet(x, y) {
                        Some(met) => positional.push(met),
                        None => return self.incompatible(&a, &b, span),
                    }
                }
                let mut named = Vec::new();
                for (name, x) in &n1 {
                    let Some((_, y)) = n2.iter().find(|(n, _)| n == name) else {
                        continue;
                    };
                    match self.meet(x, y) {
                        Some(met) => named.push((name.clone(), met)),
                        None => return self.incompatible(&a, &b, span),
                    }
                }
                let result = self.join(r1, r2, span);
                Type::Function {
                    positional,
                    named,
                    result: Box::new(result),
                }
            }
            // Two tables join conjunct by conjunct. Every pair of conjuncts that joins is a
            // type both tables are subtypes of — each side reaches it by
            // Sub-And-L through the conjunct it contributed — so their join is the
            // intersection of all such pairs. Pairs are tried rather than
            // matched on the nose because the two sides need not split their
            // domains the same way: one table's conjuncts may be single atoms where
            // the other's cover a whole class, and those still overlap. A pair
            // whose domains do not overlap contributes nothing.
            //
            // Identical tables take the unification path first, so a table
            // joined with itself keeps its refinement variables live rather
            // than grounding them.
            (Type::And(xs), Type::And(ys)) => {
                let mark = self.mark();
                if self.unify(&a, &b).is_ok() {
                    return self.resolve(&a);
                }
                self.rollback(mark);
                let (xs, ys) = (xs.clone(), ys.clone());
                if xs.len().saturating_mul(ys.len()) > MAX_TABULATION_VECTORS {
                    return self.incompatible(&a, &b, span);
                }
                let mut conjuncts: Vec<Type> = Vec::new();
                for x in &xs {
                    for y in &ys {
                        if !self.same_shape(x, y) {
                            continue;
                        }
                        // A pair that does not join contributes no conjunct — its
                        // domains do not overlap, or its results disagree —
                        // the same way a vector whose body does not check
                        // contributes none to `tabulate`, and its errors are
                        // exploratory either way.
                        let before = self.errors.len();
                        let conjunct = self.join(x.clone(), y.clone(), span);
                        if self.errors.len() > before {
                            self.errors.truncate(before);
                            continue;
                        }
                        if !conjuncts.contains(&conjunct) {
                            conjuncts.push(conjunct);
                        }
                    }
                }
                match conjuncts.len() {
                    0 => self.incompatible(&a, &b, span),
                    1 => conjuncts.pop().expect("one conjunct"),
                    _ => Type::And(conjuncts),
                }
            }
            _ => {
                let mark = self.mark();
                if self.unify(&a, &b).is_ok() {
                    self.resolve(&a)
                } else {
                    self.rollback(mark);
                    self.incompatible(&a, &b, span)
                }
            }
        }
    }

    /// Returns whether two arrows have the same parameter shape.
    ///
    /// Arity must agree and named parameters must agree by name; the parameter
    /// *types* are not consulted, since two conjuncts that overlap without
    /// coinciding still join. Anything that is not a pair of arrows has no
    /// shape in common.
    ///
    /// # Example
    ///
    /// `(int) -> int` and `(float) -> float` share a shape;
    /// `(int) -> int` and `(int, int) -> int` do not.
    fn same_shape(&self, x: &Type, y: &Type) -> bool {
        let (
            Type::Function {
                positional: p1,
                named: n1,
                ..
            },
            Type::Function {
                positional: p2,
                named: n2,
                ..
            },
        ) = (self.resolve(x), self.resolve(y))
        else {
            return false;
        };
        p1.len() == p2.len()
            && n1.len() == n2.len()
            && n1
                .iter()
                .all(|(name, _)| n2.iter().any(|(other, _)| other == name))
    }

    /// Reports two types as having no join and recovers with `Dynamic`.
    fn incompatible(&mut self, a: &Type, b: &Type, span: &Option<Span<S>>) -> Type {
        let message = format!(
            "incompatible types {} and {}",
            self.display(a),
            self.display(b)
        );
        self.error(message, span);
        Type::Dynamic
    }

    /// The meet of two types, or `None` where they have none.
    ///
    /// The dual of [`Infer::join`], for the parameters of a joined arrow:
    /// what both sides accept. Numerics meet by sort intersection, and an
    /// empty intersection is no meet at all. Arrows meet with the variance
    /// flipped again — join their parameters, meet their results — and
    /// anything else must simply unify.
    fn meet(&mut self, a: &Type, b: &Type) -> Option<Type> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        match (&a, &b) {
            (Type::Dynamic, _) | (_, Type::Dynamic) => Some(Type::Dynamic),
            (Type::Numeric(x), Type::Numeric(y)) => {
                let met = self.may_of(x).intersect(self.may_of(y));
                (!met.is_empty()).then(|| Type::ground(met))
            }
            (Type::List(x), Type::List(y)) => {
                let (x, y) = ((**x).clone(), (**y).clone());
                self.meet(&x, &y).map(|met| Type::List(Box::new(met)))
            }
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
            ) if p1.len() == p2.len() && n1.is_empty() && n2.is_empty() => {
                let (p1, r1) = (p1.clone(), *r1.clone());
                let (p2, r2) = (p2.clone(), *r2.clone());
                let mut positional = Vec::with_capacity(p1.len());
                for (x, y) in p1.iter().zip(&p2) {
                    positional.push(self.join(x.clone(), y.clone(), &None));
                }
                let result = self.meet(&r1, &r2)?;
                Some(Type::function(positional, result))
            }
            _ => {
                let mark = self.mark();
                if self.unify(&a, &b).is_ok() {
                    Some(self.resolve(&a))
                } else {
                    self.rollback(mark);
                    None
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
    ///   numeric argument's sort contained in the domain, each non-numeric
    ///   argument a subtype of it (table order is most-specific-first,
    ///   mirroring the runtime match arms);
    /// - otherwise coverage decides (`select_by_atoms`): every atom combination
    ///   of the arguments must have an accepting conjunct, and the covering conjuncts'
    ///   results join;
    /// - no coverage means the runtime has no matching arm: a rejection.
    ///
    /// Serves elimination and checking alike: `app_subtype` selects with a real
    /// Ψ frame, and `subtype` selects with a pseudo-frame built from an
    /// expected arrow's parameters.
    fn select_core(&mut self, conjuncts: &[Type], frame: &Frame<S>) -> Selection {
        // The argument sorts. Dynamic and unsolved metas may be any
        // numeric; a non-numeric argument (list, function, ...) has no sort
        // and is checked against each conjunct's domain by subtyping.
        let sorts: Vec<Option<Sort>> = frame
            .positional
            .iter()
            .map(|(argument, _)| match self.resolve(argument) {
                Type::Numeric(rep) => Some(self.may_of(&rep)),
                Type::Dynamic | Type::Meta(_) => Some(Sort::TOP),
                _ => None,
            })
            .collect();
        // Only the conjuncts shaped like this call can answer it.
        let shaped: Vec<Type> = conjuncts
            .iter()
            .filter(|conjunct| {
                matches!(conjunct, Type::Function { positional, .. }
                    if positional.len() == frame.positional.len())
            })
            .cloned()
            .collect();
        if shaped.is_empty() {
            return Selection::NoMatchingConjuncts;
        }
        // Contracts for unsolved arguments use the union of every conjunct's domain
        // at the position: the variable may still grow (an arrow's result can
        // feed back into it — see the fixed-point iteration in `subtype`), and
        // any narrower contract, such as a chosen conjunct's own domain, would
        // reject that growth. Ground arguments are already judged by
        // applicability.
        let unions = self.domain_unions(&shaped, frame.positional.len());
        // First definitely-applicable conjunct wins; its subtyping
        // commits (and is rolled back when a later position rejects it).
        for conjunct in &shaped {
            let mark = self.mark();
            if self.conjunct_applies(conjunct, &sorts, frame).is_some() {
                self.record_selection_contracts(frame, &unions);
                let Type::Function { result, .. } = conjunct else {
                    unreachable!("conjuncts are arrows");
                };
                return Selection::Selected((**result).clone());
            }
            self.rollback(mark);
        }
        // Otherwise, atom-decomposition coverage.
        self.select_by_atoms(&shaped, &sorts, frame)
    }

    /// Returns the union of the sorts for each positional parameter in
    /// `conjuncts` (which all should be function types). For each position, the
    /// loosest contract that may still grow — or `None` where any positions
    /// type is unknown or non-numeric: such a conjunct may accept non-numeric
    /// values, so the table imposes no numeric contract at that position.
    fn domain_unions(&self, conjuncts: &[Type], arity: usize) -> Vec<Option<Sort>> {
        let mut unions = vec![Some(Sort::NONE); arity];
        for conjunct in conjuncts {
            let Type::Function { positional, .. } = conjunct else {
                unreachable!("conjuncts are arrows");
            };
            for (result, ty) in unions.iter_mut().zip(positional) {
                match self.resolve(ty) {
                    Type::Numeric(rep) => {
                        let sort = self.may_of(&rep);
                        *result = result.map(|union| union.union(sort));
                    }
                    // A `Sort` is a union of numeric atoms and has no room for
                    // anything else, so a non-numeric domain leaves the
                    // position with no expressible union at all.
                    _ => *result = None,
                }
            }
        }
        unions
    }

    /// Whether the conjunct's named parameters line up with the call's named
    /// arguments, each supplied argument fitting its domain.
    ///
    /// A conjunct declares exactly the names its call must supply. Tabulation
    /// emits one conjunct per omission pattern — declaring a named parameter
    /// for calls that pass it, and leaving the name off for calls that take the
    /// default — so matching the two sets is what keeps those apart. Both
    /// directions matter: a conjunct that ignored an argument the call supplies
    /// would select on the strength of a default the call overrode, and a
    /// conjunct that declared a parameter the call omits would answer for a
    /// value the call never passes.
    fn named_matches(&mut self, named: &[(String, Type)], frame: &Frame<S>) -> bool {
        for (name, _, _) in &frame.named {
            if !named.iter().any(|(declared, _)| declared == name) {
                return false;
            }
        }
        for (name, domain) in named {
            let Some((_, argument, _)) = frame.named.iter().find(|(n, _, _)| n == name) else {
                return false;
            };
            let sort = match self.resolve(argument) {
                Type::Numeric(rep) => Some(self.may_of(&rep)),
                Type::Dynamic | Type::Meta(_) => Some(Sort::TOP),
                _ => None,
            };
            // TODO record named-argument contracts the way
            // `record_selection_contracts` does for positional ones.
            let (fits, _) = self.position_fits(argument, sort, domain);
            if !fits {
                return false;
            }
        }
        true
    }

    /// Returns the conjunct's domains as contract sorts when the conjunct applies
    /// outright — every argument's sort contained in its domain — and
    /// `None` otherwise. Numeric positions check by sort containment and
    /// report their domain sort; unknown and non-numeric domains accept by
    /// subtyping and report ⊤ (no sort contract). Named parameters are
    /// judged by `named_matches`. Subtyping solves state, so
    /// callers snapshot around the call.
    fn conjunct_applies(
        &mut self,
        conjunct: &Type,
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Option<Vec<Sort>> {
        let Type::Function {
            positional, named, ..
        } = conjunct
        else {
            unreachable!("conjuncts are arrows");
        };
        let mut domains = Vec::with_capacity(positional.len());
        for ((sort, domain), (argument, _)) in sorts.iter().zip(positional).zip(&frame.positional) {
            let (fits, domain) = self.position_fits(argument, *sort, domain);
            domains.push(domain);
            if !fits {
                return None;
            }
        }
        if !self.named_matches(named, frame) {
            return None;
        }
        Some(domains)
    }

    /// Selection by atom coverage (sorts are unions over a disjoint atom basis
    /// — Freeman's union normal form): a runtime value inhabits exactly one
    /// atom, so a call is covered when every combination of its arguments'
    /// atoms has a conjunct that accepts it; the result is the join of the
    /// covering conjuncts' results, and each argument's contract is the union
    /// of the domains that admitted it. A combination no conjunct covers means
    /// the runtime has no matching arm.
    ///
    /// What that costs an argument depends on what is still open about it:
    ///
    /// - A ground sort, and a variable whose guarantees have arrived, enumerate
    ///   atoms that *will* be passed, so an uncovered combination is a
    ///   rejection.
    /// - A variable with no guarantees yet enumerates the atoms its contract
    ///   still admits, and an uncovered combination *narrows* that contract
    ///   instead of rejecting: the atoms coverage leaves are what the argument
    ///   is then bound to. This is what keeps a relational gap — `+` has no
    ///   `(seq, seq)` conjunct — from escaping through an unsolved argument,
    ///   and it is why `fn(x) => x + x` takes a waveform rather than any
    ///   numeric.
    /// - `Dynamic`, a non-numeric argument, and a position where some
    ///   conjunct's domain is not numeric decompose into no atoms at all: any
    ///   conjunct covers them, and they are constrained per conjunct by
    ///   subtyping.
    ///
    /// Positions holding the same variable share one choice, so an aliased
    /// argument moves in lockstep and is never asked for a conjunct covering a
    /// combination it cannot produce.
    fn select_by_atoms(
        &mut self,
        conjuncts: &[Type],
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Selection {
        let arity = frame.positional.len();
        // Per conjunct: domain sorts (⊤ for unknown and non-numeric domains), the
        // non-numeric and named constraints, and the result resolved before
        // the probe's rollback.
        struct Candidate {
            domains: Vec<Sort>,
            result: Type,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut admitting: Vec<Type> = Vec::new();
        for conjunct in conjuncts {
            // The probe's own refinement variables are popped by its
            // rollback, so the result it carries out is grounded at them
            // (`resolve_refinements`); variables that predate the probe
            // survive it and stay live.
            let floor = self.refs.len() as u32;
            let mark = self.mark();
            let applies = self.conjunct_admits(conjunct, sorts, frame);
            let candidate = applies.map(|domains| {
                let Type::Function { result, .. } = conjunct else {
                    unreachable!("conjuncts are arrows");
                };
                Candidate {
                    domains,
                    result: self.resolve_refinements(&result.apply(&self.subst), floor),
                }
            });
            self.rollback(mark);
            if let Some(candidate) = candidate {
                candidates.push(candidate);
                admitting.push(conjunct.clone());
            }
        }
        if candidates.is_empty() {
            return Selection::NoApplicableConjunct;
        }
        // Contracts come from the conjuncts that could still serve this call, not
        // from every conjunct in the table: an argument the caller has already
        // pinned rules conjuncts out, and a conjunct ruled out imposes nothing. `x !=
        // 0.5` keeps only the float conjunct, so `x` is a float — reading the
        // whole table would see bool and string domains too and demand
        // nothing at all.
        let unions = self.domain_unions(&admitting, arity);
        // The unknown behind an open position, for aliasing: two positions
        // naming the same one enumerate together.
        #[derive(PartialEq)]
        enum Unknown {
            Refinement(u32),
            Meta(u32),
        }
        // How each position takes part: no atoms, atoms that will arrive,
        // or the slot of an open group whose atoms may still be narrowed.
        enum Choice {
            Wild,
            Definite,
            Open(usize),
        }
        let mut open: Vec<(Unknown, Sort)> = Vec::new();
        let mut choices: Vec<Choice> = Vec::with_capacity(arity);
        let mut definite: Vec<Sort> = vec![Sort::NONE; arity];
        for (position, ((sort, (argument, _)), unions)) in
            sorts.iter().zip(&frame.positional).zip(&unions).enumerate()
        {
            // A non-numeric argument carries no sort of its own.
            if sort.is_none() {
                choices.push(Choice::Wild);
                continue;
            }
            let (unknown, contract) = match self.resolve(argument) {
                Type::Numeric(rep) => {
                    let guarantees = self.definite_of(&rep);
                    if !guarantees.is_empty() {
                        definite[position] = guarantees;
                        choices.push(Choice::Definite);
                        continue;
                    }
                    match rep {
                        Refinement::Var(id) => (
                            Unknown::Refinement(id),
                            self.contract_of(&Refinement::Var(id)),
                        ),
                        // A ground sort always has atoms, so it was
                        // definite above.
                        Refinement::Ground(_) => unreachable!("ground sorts are non-empty"),
                    }
                }
                // Every conjunct's domain here is numeric and the conjuncts are all
                // the arms there are, so an unconstrained argument may be
                // any numeric; `record_selection_contracts` is what commits
                // it, once selection succeeds.
                Type::Meta(id) => (Unknown::Meta(id), Sort::TOP),
                // `Dynamic` passes everything and imposes nothing.
                _ => {
                    choices.push(Choice::Wild);
                    continue;
                }
            };
            // An unsolved argument is only narrowable where the table
            // imposes a numeric contract at all. Where some conjunct's domain is
            // not numeric — `==` compares bools and strings as well as
            // floats — `unions` is `None` and there is nothing to bound it
            // to, so it decomposes into no atoms. A ground argument, or one
            // whose guarantees have arrived, has already been enumerated
            // above: what it holds is known whatever the table looks like.
            if unions.is_none() {
                choices.push(Choice::Wild);
                continue;
            }
            let slot = match open.iter().position(|(known, _)| *known == unknown) {
                Some(slot) => slot,
                None => {
                    open.push((unknown, contract));
                    open.len() - 1
                }
            };
            choices.push(Choice::Open(slot));
        }
        // The enumeration slots: the open groups first (those are the ones
        // narrowing rewrites), then one per definite position.
        let mut slots: Vec<(Vec<usize>, Vec<Sort>)> = Vec::new();
        for (slot, (_, contract)) in open.iter().enumerate() {
            let positions = choices
                .iter()
                .enumerate()
                .filter(|(_, choice)| matches!(choice, Choice::Open(group) if *group == slot))
                .map(|(position, _)| position)
                .collect();
            slots.push((positions, contract.atoms().collect()));
        }
        let groups = slots.len();
        for (position, choice) in choices.iter().enumerate() {
            if matches!(choice, Choice::Definite) {
                slots.push((vec![position], definite[position].atoms().collect()));
            }
        }
        // Walks every combination of the slots' atoms — the empty product
        // is one empty combination — reporting the slot digits and the
        // atom each position takes (`None` where a position has none).
        fn combinations(
            slots: &[(Vec<usize>, Vec<Sort>)],
            arity: usize,
            mut visit: impl FnMut(&[usize], &[Option<Sort>]),
        ) {
            let mut odometer = vec![0usize; slots.len()];
            loop {
                let mut atoms: Vec<Option<Sort>> = vec![None; arity];
                for (slot, digit) in slots.iter().zip(&odometer) {
                    for position in &slot.0 {
                        atoms[*position] = Some(slot.1[*digit]);
                    }
                }
                visit(&odometer, &atoms);
                let mut slot = 0;
                loop {
                    if slot == slots.len() {
                        return;
                    }
                    if odometer[slot] + 1 < slots[slot].1.len() {
                        odometer[slot] += 1;
                        break;
                    }
                    odometer[slot] = 0;
                    slot += 1;
                }
            }
        }
        // The first candidate accepting every position's atom; table order
        // is most-specific-first, mirroring the runtime's match arms.
        fn covering(candidates: &[Candidate], atoms: &[Option<Sort>]) -> Option<usize> {
            candidates.iter().position(|candidate| {
                atoms
                    .iter()
                    .zip(&candidate.domains)
                    .all(|(atom, domain)| match atom {
                        Some(atom) => atom.is_subset(*domain),
                        None => true,
                    })
            })
        }
        // An open atom that takes part in *any* uncovered combination is
        // one the argument must not turn out to be. Dropping all of them at
        // once leaves only covered combinations behind: an uncovered one
        // would have had every open atom of its own dropped.
        let mut dropped: Vec<Vec<bool>> = slots[..groups]
            .iter()
            .map(|(_, atoms)| vec![false; atoms.len()])
            .collect();
        let mut uncovered = false;
        combinations(&slots, arity, |odometer, atoms| {
            if covering(&candidates, atoms).is_none() {
                uncovered = true;
                for (slot, dropped) in dropped.iter_mut().enumerate() {
                    dropped[odometer[slot]] = true;
                }
            }
        });
        if uncovered {
            // With nothing open there is nothing to narrow: the call has no
            // arm.
            if groups == 0 {
                return Selection::NoApplicableConjunct;
            }
            for (slot, dropped) in dropped.iter().enumerate() {
                let kept: Vec<Sort> = slots[slot]
                    .1
                    .iter()
                    .zip(dropped)
                    .filter(|(_, dropped)| !**dropped)
                    .map(|(atom, _)| *atom)
                    .collect();
                if kept.is_empty() {
                    return Selection::NoApplicableConjunct;
                }
                slots[slot].1 = kept;
            }
        }
        // Every surviving combination is covered; the result is the join of
        // the distinct conjuncts that cover them.
        let mut covers: Vec<usize> = Vec::new();
        combinations(&slots, arity, |_, atoms| {
            if let Some(index) = covering(&candidates, atoms)
                && !covers.contains(&index)
            {
                covers.push(index);
            }
        });
        let mut result: Option<Type> = None;
        for index in covers {
            let candidate = candidates[index].result.clone();
            result = Some(match result {
                None => candidate,
                Some(previous) => self.join(previous, candidate, &frame.span),
            });
        }
        let result = result.expect("at least one combination");
        // An open argument's contract is what coverage left it; everything
        // else takes the table's per-position union.
        let mut contracts = unions;
        for (positions, atoms) in &slots[..groups] {
            let narrowed = atoms
                .iter()
                .fold(Sort::NONE, |union, atom| union.union(*atom));
            for position in positions {
                contracts[*position] = Some(narrowed);
            }
        }
        self.record_selection_contracts(frame, &contracts);
        Selection::Selected(result)
    }

    /// Whether a conjunct can participate in coverage at all: non-numeric
    /// arguments must subtype their domains, named parameters must line up as
    /// in `conjunct_applies`, and each position's contract sort is reported the
    /// same way. Numeric positions are not judged here — coverage judges them
    /// atom by atom.
    fn conjunct_admits(
        &mut self,
        conjunct: &Type,
        sorts: &[Option<Sort>],
        frame: &Frame<S>,
    ) -> Option<Vec<Sort>> {
        let Type::Function {
            positional, named, ..
        } = conjunct
        else {
            unreachable!("conjuncts are arrows");
        };
        let mut domains = Vec::with_capacity(positional.len());
        for ((sort, domain), (argument, _)) in sorts.iter().zip(positional).zip(&frame.positional) {
            // Numeric against numeric is coverage's job; every other pairing
            // is judged exactly as `conjunct_applies` judges it.
            let fits = match (sort, self.resolve(domain)) {
                (Some(_), Type::Numeric(rep)) => {
                    domains.push(self.may_of(&rep));
                    true
                }
                _ => {
                    let (fits, contract) = self.position_fits(argument, *sort, domain);
                    domains.push(contract);
                    fits
                }
            };
            if !fits {
                return None;
            }
        }
        if !self.named_matches(named, frame) {
            return None;
        }
        Some(domains)
    }

    /// Whether one argument fits one domain — sort containment for
    /// numeric domains, subtyping for the rest — and the contract sort the
    /// domain imposes (⊤ for variable and non-numeric domains). Subtyping
    /// solves state; callers snapshot.
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
            // A non-numeric argument never fits a numeric domain.
            (None, Type::Numeric(_)) => (false, Sort::TOP),
            // The recovery type accepts anything and imposes nothing.
            (_, Type::Dynamic) => (true, Sort::TOP),
            // A variable domain is the conjunct's own parameter, and the conjunct's
            // result may be that same variable, so the argument has to
            // flow into it — the AS-Fun2 premise `check_frame` applies at
            // ordinary calls. Accepting without binding would leave the
            // result variable free for the caller to solve at will. The
            // domain still imposes no sort contract of its own.
            (_, Type::Meta(_) | Type::Var(_)) => {
                (self.subtype(argument, &resolved).is_ok(), Sort::TOP)
            }
            // A numeric argument has no arm at a non-numeric domain.
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
        // anything, so report it and stop: selection would only add a
        // second error saying no use of the function accepts the call.
        let mut undeclared = false;
        for (name, _, _) in &frame.named {
            let declared = conjuncts.iter().any(|conjunct| {
                matches!(conjunct, Type::Function { named, .. }
                    if named.iter().any(|(n, _)| n == name))
            });
            if !declared {
                self.error(format!("no named parameter \"{}\"", name), &frame.span);
                undeclared = true;
            }
        }
        if undeclared {
            return None;
        }
        match self.select_core(conjuncts, frame) {
            Selection::Selected(result) => Some(result),
            // Nothing has the call's shape: when the table is unambiguous
            // about its arity, report the arity mismatch the way
            // `check_frame` would.
            Selection::NoMatchingConjuncts => {
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
            Selection::NoApplicableConjunct => {
                // TODO this case is pretty long... for just better error
                // messages; re-evaluate whether or not all of this logic is
                // necessary.
                // When one argument alone rules out every conjunct, pinpoint it
                // the way `check_frame` would: report at that argument with the
                // union of the domains it failed.
                let shaped: Vec<&Type> = conjuncts
                    .iter()
                    .filter(|conjunct| {
                        matches!(conjunct, Type::Function { positional, .. }
                            if positional.len() == frame.positional.len())
                    })
                    .collect();
                // A table with exactly one conjunct of the call's shape is a
                // plain arrow as far as the call is concerned, so let
                // `check_frame` say precisely which argument is wrong. Its
                // judgment is finer than applicability's sort containment and
                // may find nothing to report, in which case the heuristics
                // below still run.
                let named_matching: Vec<&&Type> = shaped
                    .iter()
                    .filter(|conjunct| {
                        let Type::Function { named, .. } = conjunct else {
                            unreachable!("conjuncts are arrows");
                        };
                        named.len() == frame.named.len()
                            && named
                                .iter()
                                .all(|(name, _)| frame.named.iter().any(|(n, _, _)| n == name))
                    })
                    .collect();
                if let [conjunct] = &named_matching[..] {
                    let Type::Function {
                        positional, named, ..
                    } = **conjunct
                    else {
                        unreachable!("conjuncts are arrows");
                    };
                    let (positional, named) = (positional.clone(), named.clone());
                    let errors = self.errors.len();
                    self.check_frame(frame, &positional, &named);
                    if self.errors.len() > errors {
                        return None;
                    }
                }
                for (position, (argument, span)) in frame.positional.iter().enumerate() {
                    let Type::Numeric(rep) = self.resolve(argument) else {
                        continue;
                    };
                    let may = self.may_of(&rep);
                    let mut union = Sort::NONE;
                    let mut sorted = true;
                    for conjunct in &shaped {
                        let Type::Function { positional, .. } = conjunct else {
                            unreachable!("conjuncts are arrows");
                        };
                        match self.resolve(&positional[position]) {
                            Type::Numeric(rep) => union = union.union(self.may_of(&rep)),
                            _ => sorted = false,
                        }
                    }
                    // `shaped`, not every conjunct: `union` is accumulated
                    // from the shaped ones, so an empty set would leave it
                    // `NONE`, which every argument fails to intersect —
                    // reporting "expected nothing" at a position no conjunct
                    // constrains.
                    if sorted && !shaped.is_empty() && may.intersect(union).is_empty() {
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
                // "Two seqs" names a *relational* gap: each position takes a
                // seq on its own and the table simply has no conjunct taking both.
                // Where no conjunct admits one at all — `==` compares floats, bools
                // and strings — the seqs are not the relation that failed, and
                // the general message says more.
                let admits_a_seq = |position: usize| {
                    shaped.iter().any(|conjunct| {
                        let Type::Function { positional, .. } = conjunct else {
                            unreachable!("conjuncts are arrows");
                        };
                        matches!(self.resolve(&positional[position]), Type::Numeric(rep)
                            if !self.may_of(&rep).intersect(Sort::SEQ).is_empty())
                    })
                };
                let both_seqs =
                    frame.positional.len() == 2 && seqs == 2 && admits_a_seq(0) && admits_a_seq(1);
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
    fn record_selection_contracts(&mut self, frame: &Frame<S>, domains: &[Option<Sort>]) {
        for ((argument, span), domain) in frame.positional.iter().zip(domains) {
            // No contract where some conjunct's domain is non-numeric or
            // unknown: that conjunct may accept non-numeric values, so the
            // table demands nothing of the argument.
            let Some(domain) = domain else { continue };
            let found = match self.resolve(argument) {
                Type::Numeric(rep) => rep,
                // An unconstrained argument commits to numeric here — every
                // conjunct's domain at this position is numeric, and the
                // table's conjuncts are all the runtime arms there are — so it
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
        // A call of the wrong arity has no pairing worth judging: matching
        // the arguments off in order would report a mismatch for every
        // position the shift moved. Rule AT-Lam2 binds `Dynamic` for the
        // same reason. Named arguments are judged either way, since which
        // names a call passes does not depend on how many positions it got
        // right.
        let arity = frame.positional.len() == positional.len();
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
        // the rest, so conjunct selection sees the sibling arguments' flows
        // first — e.g. a fold's seed constrains the accumulator before
        // the fold function's conjunct is chosen.
        let table = |infer: &Self, ty: &Type| match infer.resolve(ty) {
            Type::And(_) => true,
            Type::Forall(_, body) => matches!(*body, Type::And(_)),
            _ => false,
        };
        if arity {
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
                    // A type query about this name wants what the context
                    // holds, not the residual after Ψ: on the `f` of
                    // `f(1)` the residual is the call's result.
                    self.probe_type(&expr.span, &ty);
                    self.app_subtype(psi, ty, Some(name))
                }
                Some((_, ContextEntry::Builtin(builtin))) => {
                    let builtin = builtin.clone();
                    let ty = signatures::signature(&builtin).unwrap_or(Type::Dynamic);
                    self.probe_type(&expr.span, &ty);
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
                let errors = self.errors.len();
                let named_types: Vec<(String, Type)> = named
                    .iter()
                    .map(|(name, default)| {
                        let default_ty = self.infer(context, &mut Vec::new(), default);
                        let parameter = self.default_parameter(&default_ty, &default.span);
                        (name.clone(), parameter)
                    })
                    .collect();
                let depth = context.len();
                let unapplied = psi.is_empty();
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
                let explored = self.errors.len();
                context.truncate(depth);
                // ABS at every abstraction: an unapplied lambda is tabulated
                // wherever it stands, not only as a definition's right-hand
                // side or a conjunct's body. A lambda reached through a `let`
                // chain — `fn(a) => let x = ... in fn(w) => ...` — is
                // neither, and it is a shape the library is written in.
                if unapplied && let Some(table) = self.tabulate(context, expr, &result, None) {
                    self.errors.drain(errors..explored);
                    return table;
                }
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
                    // The recovery type projects to itself, so one
                    // reported error does not cascade into more.
                    Type::Dynamic => Type::Dynamic,
                    // An unknown does not: a projection has to name a module
                    // the checker can look the name up in, and the type here
                    // says nothing about which module — or whether it is one.
                    // Rejecting is what makes every surviving projection
                    // statically checked, at the cost of functions taking
                    // modules as parameters. The type reaches the projection
                    // wherever it is tracked at all (a `let`, a directly
                    // applied lambda, a list, a tuple element), so what this
                    // rules out is the unannotated parameter, where it
                    // genuinely is not known.
                    Type::Meta(_) => {
                        let message =
                            format!("cannot project '{}' from a value of unknown type", name);
                        self.error(message, &expr.span);
                        Type::Dynamic
                    }
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
                self.probe_type(&expr.span, &ty);
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
        memo: &mut ModuleMemo,
    ) -> Vec<(String, ContextEntry)>
    where
        F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    {
        let mut own = Vec::new();
        for source_binding in bindings {
            match &source_binding.binding {
                Binding::Open(path) => match resolve(path) {
                    Ok(module) => {
                        let exports =
                            self.module_exports(resolve, module, memo, path, &source_binding.span);
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
                    // Evaluation refuses this, so the checker must too: the
                    // parser only produces an empty path when the path
                    // failed to parse, and it reports that itself.
                    let Some(name) = path.last() else {
                        self.error(
                            "`use` requires a module path".to_string(),
                            &source_binding.span,
                        );
                        continue;
                    };
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
                    let exports =
                        self.module_exports(resolve, module, memo, path, &source_binding.span);
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
        memo: &mut ModuleMemo,
        path: &[String],
        span: &Option<Span<S>>,
    ) -> Vec<(String, ContextEntry)>
    where
        F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    {
        let key = module_key(module);
        match memo.get(&key) {
            Some(Some(exports)) => return exports.clone(),
            // Reached while still building: the module opens something that
            // opens it back. The sentinel goes in *before* recursing so the
            // second visit stops here rather than recurring forever.
            Some(None) => {
                let message = format!("module '{}' is opened from itself", path.join("."));
                self.error(message, span);
                return Vec::new();
            }
            None => {}
        }
        memo.insert(key, None);
        let mut module_context = Vec::new();
        let exports = self.build_context(resolve, module, &mut module_context, memo);
        memo.insert(key, Some(exports.clone()));
        exports
    }
}

/// Coalesces tabulated conjuncts: two conjuncts that differ at just one numeric
/// domain position and agree on the result merge into one conjunct with the
/// union domain there — e.g. `({I}) -> float ∧ ({NonInt}) -> float` becomes
/// `(float) -> float`. Purely a simplification: selection reads the merged
/// table the same way, and displays stay legible.
fn merge_conjuncts(mut conjuncts: Vec<Type>) -> Vec<Type> {
    'restart: loop {
        for i in 0..conjuncts.len() {
            for j in (i + 1)..conjuncts.len() {
                if let Some(merged) = merge_pair(&conjuncts[i], &conjuncts[j]) {
                    conjuncts[i] = merged;
                    conjuncts.remove(j);
                    continue 'restart;
                }
            }
        }
        return conjuncts;
    }
}

/// Returns the union of conjuncts `a` and `b` when they agree everywhere except
/// at most one numeric ground domain position (positional or named).
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
    use crate::expr::ErrorKind;
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
                function: BuiltInFn(Rc::new(|_| Err(Error::internal_here("stub")))),
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
            |_: &[String]| Err(Error::types_here("no modules".to_string())),
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
    // against `(a) -> b`, and the int conjunct covers it.
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
        // conjuncts), so seq-ness survives to the following `\`.
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
    // intersections of arrows, one conjunct per applicable atom vector.
    #[test]
    fn tabulated_definitions() {
        // Result precision at a direct call: double(2) is an int, and an
        // int is definitely not a seq.
        assert_errors(
            "let double = fn(x) => x + x in double(2) \\ 1",
            &["expected seq, found int"],
        );
        // The missing (seq, seq) arm of + is relational and survives the
        // definition boundary: double has no seq conjunct at all.
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
        // union of its conjuncts), so a mixed list stays silent...
        assert_clean("map(fn(x) => x * 0.5, [time, 2])");
        // ...and a seq element is fine where the body threads seqs...
        assert_clean("map(fn(x) => x * 2, [seq(0)(1)])");
        // ...but errors where no conjunct accepts one: x + x has no seq conjunct.
        // The list is checked before the table (see `check_frame`), so
        // the message shows the seq flowing into the expected arrow
        // against the conjuncts the function actually has.
        assert_errors(
            "map(fn(x) => x + x, [seq(0)(1)])",
            &[
                "expected (seq) -> ?a, found (int) -> int ∧ (float) -> float ∧ (waveform) -> waveform",
            ],
        );
    }

    // Mixed parameter lists: numeric parameters tabulate while non-numeric ones
    // ride along as quantified unknowns.
    #[test]
    fn tabulated_mixed_parameters() {
        // The fold's accumulator conjunct is precise: an int seed selects the
        // int conjunct, so the result is definitely not a seq.
        assert_errors(
            "reduce(fn(acc, x) => acc + 1, 0, [1, 2]) \\ 1",
            &["expected seq, found int"],
        );
        // The seed is checked before the table, so a float seed selects the
        // float conjunct rather than erroring against the int conjunct.
        assert_clean("reduce(fn(acc, x) => acc + 1, 0.5, [1, 2])");
        // A non-numeric domain solved by the body (xs used as a list)
        // participates in selection at direct calls.
        assert_clean("let pick = fn(n, xs) => nth(n, xs) in pick(1, [1, 2])");
        assert_errors(
            "let pick = fn(n, xs) => nth(n, xs) in pick(0.5, [1, 2])",
            &["expected int, found float"],
        );
    }

    // A non-numeric domain the body leaves *unsolved* stays a bare variable in
    // every conjunct, and that variable is also the conjunct's result.
    // Selection binds it like any other parameter, so the call's result is the
    // argument's type rather than an unknown the caller may solve at will.
    #[test]
    fn variable_domains_bind_their_argument() {
        // f : ('a, int) -> 'a ∧ ('b, float) -> 'b — `x` is only passed
        // through, so no conjunct constrains it.
        let f = "let f = fn(x, n) => if n > 0 then x else x in ";
        assert_clean(&format!("{}nth(f(1, 1), [1, 2])", f));
        assert_errors(
            &format!("{}nth(f(\"s\", 1), [1, 2])", f),
            &["expected int, found string"],
        );
        // A numeric argument binds too: the result is the seq that went in.
        assert_errors(
            &format!("{}nth(f(seq(0)(1), 1), [1, 2])", f),
            &["expected int, found seq"],
        );
        // An omitted named argument puts the same shape on the coverage
        // path, where conjuncts are candidates rather than applying outright.
        let g = "let g = fn(x, k = 2) => x in ";
        assert_clean(&format!("{}nth(g(1), [1, 2])", g));
        assert_errors(
            &format!("{}nth(g(\"s\"), [1, 2])", g),
            &["expected int, found string"],
        );
        assert_errors(
            &format!("{}<[g(true)]>", g),
            &["expected [seq], found [bool]"],
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
        // ...while the body's missing seq conjunct still rejects a seq.
        assert_errors(
            &format!("{}f(x = seq(0)(1))", f),
            &["no use of f accepts (x = seq)"],
        );
        // An omitted argument takes the default; the join of the table
        // covers it, so no single conjunct's result is claimed.
        assert_clean(&format!("{}f()", f));
        // Result precision flows through a supplied named argument.
        assert_errors(&format!("{}f(x = 2) \\ 1", f), &["expected seq, found int"]);
        // A default the body cannot accept leaves the omitted conjunct out of the
        // table, and a call may always omit it, so the definition is what is
        // reported.
        assert_errors(
            "fn(x = 100) => x \\ 1",
            &["default value for \"x\" cannot be used in the body"],
        );
    }

    #[test]
    fn a_conflict_on_a_nested_sort_is_reported_as_one() {
        // A conflict on a numeric nested inside two arrows cannot be shown by
        // naming them: a variable renders as the guarantees it has collected,
        // so the two sides read as though they should match.
        assert_errors(
            "let h = fn(v) => v | fin(4) in \
             let f = fn(x, k = h) => k in nth(f(1, k = cos), [1, 2])",
            &["expected int, found (waveform) -> waveform"],
        );
        // Where either side *is* a numeric, its name states the sort and the
        // ordinary message says more.
        assert_errors("nth(time, [1, 2, 3])", &["expected int, found waveform"]);
        assert_errors(
            "let f = fn(k = time) => k in nth(f(), [1, 2, 3])",
            &["expected int, found waveform"],
        );
        // A mismatch that is not about sorts at all is untouched.
        assert_errors(
            "reduce(fn(acc, x) => fixed, sqrt, [1])",
            &["expected ([float]) -> waveform, found (float) -> float"],
        );
    }

    #[test]
    fn an_arrow_met_by_a_meta_varies_like_an_arrow() {
        // Arrow unification (Xie and Oliveira, Fig. 14): the meta becomes an
        // arrow of fresh parts, so a fold whose seed and step are different
        // functions settles on the arrow both fit rather than demanding one
        // of them exactly.
        assert_clean("reduce(fn(acc, x) => cos, sqrt, [1])");
        assert_clean("unfold(fn(v) => cos, sqrt, 2)");
        // Through a structure, too, where the leaf is the arrow.
        assert_clean("reduce(fn(acc, x) => [cos], [sqrt], [1])");
        assert_clean("reduce(fn(acc, x) => (cos, 1), (sqrt, 2), [1])");
        // Parameters meet and results join, the same answer `join` gives two
        // arrows: `sqrt` takes floats, so the fold's does too.
        assert_clean("reduce(fn(acc, x) => cos, sqrt, [1])(0.5)");
        assert_errors(
            "reduce(fn(acc, x) => cos, sqrt, [1])(time)",
            &["expected float, found waveform"],
        );
        // Arrows with no shared parameter shape still have nothing to meet.
        assert_errors(
            "reduce(fn(acc, x) => fixed, sqrt, [1])",
            &["expected ([float]) -> waveform, found (float) -> float"],
        );
        // A function-valued named default still pins its parameter, and a
        // definition-bound one is quantified, so the pin has to look through
        // the quantifier to see the arrow.
        assert_errors(
            "let g = fn(v) => 1 in let h = fn(v) => time in \
             let f = fn(k = g) => k(0) in nth(f(k = h), [1, 2, 3])",
            &["expected int, found waveform"],
        );
        assert_clean("let g = fn(v) => 1 in let f = fn(k = g) => k(0) in nth(f(), [1, 2, 3])");
    }

    #[test]
    fn a_structure_met_by_a_meta_keeps_its_leaves_open() {
        // A list of ints flowing into a polymorphic parameter must not pin
        // the element's sort: a fold's seed is only its starting value, so a
        // step returning waveforms has to be able to widen it.
        assert_clean("unfold(fn(v) => [time], [1, 2], 2)");
        assert_clean("reduce(fn(acc, x) => [time], [1, 2], [2])");
        assert_clean("reduce(fn(acc, x) => (time, 1), (1, 2), [2])");
        // The widened leaf is what the result carries, not the seed's sort.
        assert_errors(
            "nth(nth(0, reduce(fn(acc, x) => [time], [1, 2], [2])), [1, 2, 3])",
            &["expected int, found int or waveform"],
        );
        // Only numeric leaves open up: the structure itself, and leaves of
        // other kinds, still have to agree.
        assert_errors(
            "unfold(fn(v) => [true], [1, 2], 2)",
            &["expected [bool], found [int]"],
        );
        assert_errors(
            "unfold(fn(v) => time, [1, 2], 2)",
            &["expected waveform, found [int]"],
        );
        // The supplied conjunct carries the caller's element type through to the
        // result, so this is caught where the result is used rather than at
        // the argument.
        assert_errors(
            "let f = fn(k = [1, 2]) => nth(0, k) in nth(f(k = [time]), [1, 2])",
            &["expected int, found waveform"],
        );
        // The omitted conjunct keeps the default's type, so a call that leaves
        // the argument out still gets the default's result.
        assert_errors(
            "let f = fn(k = [1, 2]) => nth(0, k) in f() \\ 1",
            &["expected seq, found int"],
        );
    }

    // A call that omits a named argument gets the value the default
    // supplies, so tabulation gives every named parameter an "omitted"
    // choice: a conjunct that leaves the name off and checks the body at the
    // default's own type. Selection matches the conjunct's names against the
    // call's, so the omitted conjunct answers only for calls that omit.
    #[test]
    fn omitted_named_arguments_take_their_default() {
        // The default's sort decides the result, not the first conjunct's.
        assert_errors(
            "let f = fn(k = time) => k in nth(f(), [1, 2, 3])",
            &["expected int, found waveform"],
        );
        assert_errors(
            "let f = fn(k = 0.5) => k in nth(f(), [1, 2, 3])",
            &["expected int, found float"],
        );
        // An int default still gets the precise int result.
        assert_clean("let f = fn(k = 2) => k in nth(f(), [1, 2, 3])");
        assert_clean("let f = fn(x, k = 2) => x * k in nth(f(3), [1, 2, 3])");
        // Each named parameter takes its own default, so a waveform in any
        // of them carries through.
        assert_errors(
            "let f = fn(a, b = 1, c = time) => a + b + c in nth(f(1), [1, 2, 3])",
            &["expected int, found waveform"],
        );
        // A supplied argument overrides the default and selects the conjunct
        // that declares the name — the omitted conjunct must not answer for it.
        assert_clean("let f = fn(k = 2) => k in nth(f(k = 1), [1, 2, 3])");
        assert_errors(
            "let f = fn(k = 2) => k in nth(f(k = time), [1, 2, 3])",
            &["expected int, found waveform"],
        );
        // A supplied argument is judged by the body's use of the parameter,
        // not by the default: `nth`'s element type is free, so a list of
        // bools is a list `nth` can index. The index being past the end is
        // an evaluation fault, which no sort rules out.
        assert_clean("let f = fn(x, k = [1, 2]) => nth(x, k) in f(1, k = [true])");
        assert_clean("let f = fn(x, k = [1, 2]) => nth(x, k) in f(1)");
    }

    // append is exactly the waveform primitive — two waveforms, end to
    // end — and lists join with concat.
    #[test]
    fn append_takes_waveforms_and_concat_takes_lists() {
        assert_clean("append(time * 0.5 | fin(1), time | fin(1))");
        // Floats promote to waveforms, as everywhere else.
        assert_clean("append(time, 1) * 2");
        assert_clean("concat([[1, 2], [3, 4]])");
        // The fold idiom in std's zip.
        assert_clean("reduce(fn(acc, i) => concat([acc, [i]]), [], [1, 2])");
        // Seqs and lists have no arm.
        assert_errors("append(seq(0)(1), time)", &["expected waveform, found seq"]);
        assert_errors("append([1], time)", &["expected waveform, found [int]"]);
    }

    // Freeman and Pfenning's ABS applies at every abstraction, so a
    // curried definition is tabulated at each arrow rather than only at the
    // outermost. Without that the inner parameter stays an unsolved
    // variable and selection falls back on its wildcard path, which reads
    // the first conjunct's result whatever the argument turns out to be.
    #[test]
    fn curried_definitions_tabulate_at_every_arrow() {
        let add = "let add = fn(a) => fn(b) => a + b in ";
        // Each arrow carries its own table, so the result tracks *both*
        // arguments rather than just the first.
        assert_errors(
            &format!("{}nth(add(1)(time), [1, 2, 3])", add),
            &["expected int, found waveform"],
        );
        assert_clean(&format!("{}nth(add(1)(2), [1, 2, 3])", add));
        // The relational gap survives the currying: `+` has no (seq, seq)
        // conjunct, and partial application does not lose that.
        assert_clean(&format!("{}add(seq(0)(1))", add));
        assert_errors(
            &format!("{}add(seq(0)(1))(seq(0)(2))", add),
            &["expected waveform, found seq"],
        );
        // A named parameter on the inner lambda tabulates the same way.
        let named = "let f = fn(a) => fn(k = 2) => a + k in ";
        assert_clean(&format!("{}nth(f(1)(), [1, 2, 3])", named));
        assert_errors(
            &format!("{}nth(f(1)(k = time), [1, 2, 3])", named),
            &["expected int, found waveform"],
        );
    }

    // Defaults are inferred once, outside the vectors, so an error in one is
    // reported even when the conjuncts themselves check out — the base pass's
    // summary is dropped in favor of the conjuncts, and a default depends on no
    // parameter, so it is not part of that summary.
    #[test]
    fn a_default_that_does_not_check_is_reported_at_the_definition() {
        for wrapper in ["let f = {} in 0", "let f = {} in f()", "{}"] {
            let program = wrapper.replace("{}", "fn(k = <[1, 2]>) => 1");
            assert_errors(&program, &["expected [seq], found [int]"]);
        }
    }

    // Beyond 4^k the two-point unseq/seq split still tabulates, keeping
    // seq relationality for wide parameter lists like the synths'.
    #[test]
    fn coarse_tabulation_beyond_the_cap() {
        let f = "let f = fn(a, b, c, d, e) => a + b + c + d + e in ";
        // A single seq threads through the fold of + conjuncts...
        assert_clean(&format!("{}f(1, 2, 3, seq(0)(4), 5) \\ 1", f));
        // ...but two seqs have no conjunct, even though each position alone
        // admits one.
        assert_errors(
            &format!("{}f(1, 2, 3, seq(0)(4), seq(0)(5))", f),
            &["no use of f accepts (int, int, int, seq, seq)"],
        );
    }

    // The checking policy: ground sorts judged by containment, selection
    // by atom coverage, unsolved-variable flows recorded and deferred.
    // Equality compares by value on the three kinds the runtime matches,
    // so anything it cannot take apart — a non-constant waveform, a seq, a
    // list, a function — has no arm rather than comparing false.
    #[test]
    fn equality_is_not_structural() {
        assert_clean("1 == 2");
        assert_clean("0.5 == 1");
        assert_clean("true != false");
        assert_clean("\"a\" == \"b\"");
        // A constant compares as the number it is, so the numeric conjunct is
        // `float`: a non-constant waveform has no arm.
        assert_errors("time == 1", &["no use of == accepts (waveform, int)"]);
        assert_errors(
            "[time] == [time]",
            &["no use of == accepts ([waveform], [waveform])"],
        );
        assert_errors(
            "sqrt == sqrt",
            &["no use of == accepts ((float) -> float, (float) -> float)"],
        );
        // Both sides must be the same kind.
        assert_errors("1 == true", &["no use of == accepts (int, bool)"]);
        // Two seqs get the general message, not the relational one: "cannot
        // combine two seqs" is for a table that takes a seq on either side
        // and simply has no conjunct taking both, which is `+` and not `==`.
        let seq = "(time | seq(time - 1))";
        assert_errors(
            &format!("{} == {}", seq, seq),
            &["no use of == accepts (seq, seq)"],
        );
        assert_errors(
            &format!("{} + {}", seq, seq),
            &["cannot combine two seqs with +"],
        );
    }

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
        // Atom coverage: no single conjunct contains waveform-or-seq, but the
        // waveform and seq atoms are covered by different conjuncts.
        assert_clean("(if true then time else seq(0)(1)) * 1");
        // A definitely-uncovered atom still rejects.
        assert_errors("seq(0)(1) + seq(0)(2)", &["cannot combine two seqs with +"]);
        // Unconstrained parameters defer: the base pass records contracts
        // instead of judging ⊤, and the tabulated conjuncts judge per atom.
        assert_clean("let f = fn(x) => x + 1 in f(time)");
        // Dynamic passes without imposing or cascading.
        assert_clean("debug(1) + 1");
        // A guarantee already seen is judged, deferred or not: the default
        // flows into x before the body's seq contract, and the conjunct it
        // would have made is the one tabulation drops — which is reported,
        // since no call can avoid omitting.
        assert_errors(
            "fn(x = 100) => x \\ 1",
            &["default value for \"x\" cannot be used in the body"],
        );
    }

    // A position where some conjunct's domain is non-numeric or unknown imposes
    // no numeric contract: a fold whose combiner ignores its element
    // parameter leaves the list's element type polymorphic (std's len),
    // so the same list's elements can still be used as waveforms.
    #[test]
    fn non_numeric_domains_impose_no_numeric_contract() {
        assert_clean(
            "let count = fn(xs) => reduce(fn(acc, x) => acc + 1, 0, xs) in \
             let f = fn(xs) => (count(xs), map(fn(x) => sine(x, 0), xs)) in \
             f([440])",
        );
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
    // conjunct — the float seed's doubled results stay floats, not rejections.
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

    // Coverage (`select_by_atoms`): when no single conjunct contains an
    // argument's sort, its atoms decompose and every combination must find its
    // own conjunct.
    #[test]
    fn coverage_selection() {
        // {W,S} spans the waveform and seq conjuncts of `*`: accepted, and the
        // covering conjuncts' results join — the result is waveform-or-seq, not
        // one conjunct's claim, as `\`'s rejection then shows. (It may still be
        // a seq, so no single argument is pinpointed.)
        assert_clean("(if true then time else seq(0)(1)) * 2");
        assert_errors(
            "((if true then time else seq(0)(1)) * 2) \\ 1",
            &["no use of \\ accepts (waveform or seq, int)"],
        );
        // Each argument alone is covered, but the (seq, seq) combination
        // has no conjunct: coverage judges combinations, not positions.
        assert_errors(
            "(if true then 1 else seq(0)(1)) + (if true then 2 else seq(0)(2))",
            &["no use of + accepts (int or seq, int or seq)"],
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
            |_: &[String]| Err(Error::types_here("no modules".to_string())),
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
                Err(Error::types_here(format!("no module {:?}", path)))
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
                Err(Error::types_here(format!("no module {:?}", path)))
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

        // A projection needs the checker to know which module, so an
        // unknown is rejected rather than trusted. This is the one shape
        // the rule costs: a function over modules.
        let expr = parse_program::<u32, _>("let f = fn(q) => q.two in f(b)", ()).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert_eq!(
            messages(&errors),
            ["cannot project 'two' from a value of unknown type"]
        );

        // Everywhere the type is tracked, the name is looked up — so a
        // misspelling is a static error there too, which is the point of the
        // rule.
        for (text, expected) in [
            ("let n = b in n.two", None),
            (
                "let n = b in n.three",
                Some("Module has no binding 'three'"),
            ),
            ("(fn(q) => q.two)(b)", None),
            (
                "(fn(q) => q.three)(b)",
                Some("Module has no binding 'three'"),
            ),
            ("nth(0, [b]).two", None),
            ("let (u, v) = (b, b) in u.two", None),
        ] {
            let expr = parse_program::<u32, _>(text, ()).unwrap();
            let errors = check_program(resolve, &bindings, &expr, None);
            match expected {
                None => assert!(errors.is_empty(), "{}: got {:?}", text, messages(&errors)),
                Some(message) => assert_eq!(messages(&errors), [message], "for {}", text),
            }
        }
    }

    #[test]
    fn cannot_apply_non_function() {
        assert_errors("1(2)", &["cannot apply a value of type int"]);
    }

    // Branches join pointwise with the variance: a function usable as
    // either accepts only what both accept and returns what either returns.
    // What must not happen is taking one branch's type wholesale, which
    // would hand the other's value a domain it does not accept.
    #[test]
    fn an_intersection_passed_to_a_meta_binds_whole() {
        // `g` is an intersection; a polymorphic parameter has nothing to
        // select on, so it must bind all of it. Committing to the first
        // conjunct would leave only `(int) -> int` behind.
        assert_clean("let g = fn(v) => v + 1 in let id = fn(y) => y in id(g)(time)");
        assert_clean("let g = fn(v) => v + 1 in nth(1, [g, g])(time)");
        assert_clean("let g = fn(v) => v + 1 in let box = fn(y) => [y] in nth(0, box(g))(time)");
        // Sub-And-R runs before Sub-And-L, so an intersection meeting an
        // intersection may use a different conjunct for each obligation.
        assert_clean("let h = fn(v) => v | fin(4) in let f = fn(x, k = h) => 1 in f(2, k = h)");
        // The extra conjuncts are real, not just carried: each is usable.
        assert_clean("let g = fn(v) => v + 1 in let id = fn(y) => y in id(g)(1)");
        assert_clean("let g = fn(v) => v + 1 in let id = fn(y) => y in id(g)(0.5)");
        // And a genuine mismatch is still rejected.
        assert_errors(
            "let g = fn(v) => v + 1 in let id = fn(y) => y in id(g)(\"s\")",
            &["no use of id accepts (string)"],
        );
    }

    #[test]
    fn branches_join_two_tables_row_by_row() {
        let tables = "let g = fn(v) => v + 1 in let h = fn(v) => v | fin(4) in ";
        // `h` takes waveforms and seqs, `g` takes anything; the branch offers
        // a conjunct wherever the two overlap, at the join of their results.
        assert_clean(&format!("{}(if true then g else h)(time)", tables));
        assert_clean(&format!("{}(if true then g else h)(0.5)", tables));
        // The conjuncts really are per-domain: at an int both are possible, so the
        // result is either's.
        assert_errors(
            &format!("{}nth((if true then g else h)(1), [1, 2, 3])", tables),
            &["expected int, found int or waveform"],
        );
        // Tables that differ in one conjunct join there and agree elsewhere,
        // rather than being reported wholly incompatible.
        assert_clean(
            "let mk = fn(h) => h in let u = fn(k = 1) => k in let v = fn(k = time) => k in \
             if true then mk(u) else mk(v)",
        );
        // A table joined with itself takes the unification path, so its
        // variables stay live instead of grounding.
        assert_clean("let g = fn(v) => v + 1 in (if true then g else g)(time)");
        // Rows that share no domain leave nothing to offer.
        assert_errors(
            "let g = fn(v) => v + 1 in let b = fn(v) => true in if true then g else b",
            &[
                "incompatible types (int) -> int ∧ (float) -> float ∧ (waveform) -> waveform ∧ (seq) -> seq and (?a) -> bool",
            ],
        );
    }

    #[test]
    fn branches_join_arrows_with_the_variance() {
        // sine takes waveforms and log takes floats; the branch takes what
        // both take and returns what either returns.
        assert_clean("(if false then sine else log)(0.5, 0.5)");
        // So a waveform argument is rejected — at the argument, which only
        // the joined domain can point at.
        assert_errors(
            "(if false then sine else log)(time, 0)",
            &["expected float, found waveform"],
        );
        // The join is symmetric.
        assert_clean("(if false then log else sine)(0.5, 0.5)");
        // An intersection joins with itself, which needs `unify` to know
        // about intersections at all.
        assert_clean("let g = fn(v) => v + 1 in nth(1, [g, g])(1)");
        // Lists join their elements, and agreeing branches are unchanged.
        assert_clean("nth(0, [1, time])");
        assert_clean("(if false then sqrt else exp)(4)");
        assert_clean("(if false then append else reset)(time, time)");
        // Two arrows with no common domain still have no join.
        assert_errors(
            "if false then sqrt else fixed",
            &["incompatible types (float) -> float and ([float]) -> waveform"],
        );
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
    /// Whether a runtime failure is one the sort lattice cannot see.
    ///
    /// The runtime says which it is, so this asks rather than matching on
    /// the message: a builtin marks the errors the lattice cannot judge, and
    /// everything else it reports is a sort or arity the checker was meant to
    /// rule out.
    /// The default is the safe one — a error nobody has classified counts as
    /// a checker failure, so adding a builtin cannot quietly widen what the
    /// harness overlooks.
    fn declared_residue<S>(error: &Error<S>) -> bool {
        error.kind() == ErrorKind::Eval
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
    /// function applied once per table conjunct at representative arguments — the
    /// tabulated analog of the builtin conformance test, checking result sorts
    /// against the conjuncts. Resolves modules for the differential harness; a
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
            .ok_or_else(|| Error::types_here(format!("no module {}", key)))
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
                    declared_residue(error),
                    "soundness: module {} is clean but evaluates to: {}",
                    path,
                    error.message()
                );
            }
            calls += 1;
        }
        // Every exported function applied once per all-numeric table conjunct.
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
                let conjuncts: Vec<&Type> = match body {
                    Type::And(conjuncts) => conjuncts.iter().collect(),
                    other => vec![other],
                };
                for conjunct in conjuncts {
                    let Type::Function {
                        positional, result, ..
                    } = conjunct
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
                    Err(error) if declared_residue(error) => residue += 1,
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
                                "{:?}: runtime sort {} outside the conjunct's declared {}",
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
                                "{:?}: runtime sort {} outside the conjunct's declared {}",
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
        // The harness must be exercising real applied conjuncts.
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
                .ok_or_else(|| Error::types_here(format!("no module {}", key)))
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
    fn a_default_that_does_not_work_is_reported() {
        // Every named argument may be left out, so the conjunct that omits one is
        // the only conjunct some calls can reach. When the default does not check,
        // that conjunct is dropped and the table quietly keeps the conjuncts that need
        // the argument supplied — which `check_frame` then lets an omitting
        // call through, since a named parameter is optional by construction.
        // The definition is where this can be said.
        assert_errors(
            "fn(x = 100) => x \\ 1",
            &["default value for \"x\" cannot be used in the body"],
        );
        // The call that would have used it is the one that crashed:
        // `Expected seq as first argument to \\, got 100`.
        assert_errors(
            "let f = fn(x = 100) => x \\ 1 in f()",
            &["default value for \"x\" cannot be used in the body"],
        );
        // A default the body can use is fine, supplied or not.
        assert_clean("let f = fn(x = 100) => x + 1 in f()");
        assert_clean("let f = fn(x = 100) => x + 1 in f(x = time)");
        // And a conjunct dropped for an *atom* is not reported: a caller can
        // simply not pass a seq there.
        assert_clean("let f = fn(x = 1) => sqrt(x) in f(x = 4)");
    }

    #[test]
    fn a_chord_rejects_an_instrument_it_cannot_mix() {
        // std's chord helpers in miniature — scale each note of a triad, mix
        // the result, place it in time — written without the `unseq()` those
        // helpers carry, which is the shape the checker must reject.
        //
        // `{...}` mixes waveforms, so an instrument handing back a seq makes
        // the mix fail at run time. Nothing here says which of `amp`'s conjuncts
        // the mapped element takes, so it summarises to everything `amp`
        // covers and the mix rejects it. Two details are load-bearing and the
        // gap does not show without either: the triad must be tabulated, so
        // that selecting a conjunct with an unsolved element takes the coverage
        // join, and the `| seq(time - dur)` tail must be present, so that
        // every conjunct of the innermost lambda fails and the unions base-pass
        // summary is what reports.
        let parts = "let amp = fn(a) => fn(w) => a * w in \
                     let triad = fn(root) => fn(fw) => [fw(root), fw(root + 4), fw(root + 7)] in ";
        assert_errors(
            &format!(
                "{}fn(key) => fn(inst) => fn(dur) => \
                 {{map(amp(0.4), triad(key)(fn(freq) => inst(dur, freq)))}} | seq(time - dur)",
                parts
            ),
            &["expected waveform, found waveform or seq"],
        );
        // The body they have now says what the mix needs before the
        // element's sort is chosen, and tabulates instead.
        assert_clean(&format!(
            "{}fn(key) => fn(inst) => fn(dur) => \
             {{map(amp(0.4), triad(key)(fn(freq) => inst(dur, freq) | unseq()))}} | seq(time - dur)",
            parts
        ));
        // And it still turns away an instrument that does not return a seq,
        // which would fail in `unseq` instead.
        assert_errors(
            &format!(
                "{}let chord = fn(key) => fn(inst) => fn(dur) => \
                 {{map(amp(0.4), triad(key)(fn(freq) => inst(dur, freq) | unseq()))}} | seq(time - dur) \
                 in chord(60)(fn(d, f) => f * time)(1)",
                parts
            ),
            &[
                "expected (numeric, int) -> seq, found ('a, int) -> waveform ∧ ('b, float) -> waveform ∧ ('c, waveform) -> waveform ∧ ('d, seq) -> seq",
            ],
        );
    }

    #[test]
    fn a_sibling_argument_narrows_the_table() {
        // `!=` compares floats, bools or strings. A float on one side leaves
        // only the float conjunct, so the other side is a float — reading the
        // whole table would see the bool and string domains too and demand
        // nothing, letting a seq through to a runtime failure.
        assert_errors(
            "nth(fn(x) => (x != 0.5), [1, 2])",
            &["expected int, found (float) -> bool"],
        );
        assert_errors(
            "let ap = fn(f, v) => f(v) in ap(fn(x) => (x != 0.5), (time | seq(time - 1)))",
            &["expected float, found seq"],
        );
        assert_clean("let ap = fn(f, v) => f(v) in ap(fn(x) => (x != 0.5), 1.5)");
        // Two unknowns narrow nothing, so the table demands nothing — the
        // residue N12 records.
        assert_clean("let f = fn(x) => fn(y) => 1 in f(1)(2)");
    }

    #[test]
    fn a_named_default_does_not_pin_its_parameter() {
        // A named parameter's type comes from the body's use of it. The
        // default is only what a call that omits the argument gets, so it
        // decides the omitted conjunct and nothing else.
        assert_clean("let f = fn(k = true) => 1 in f(k = 0.5)");
        assert_clean("let f = fn(k = true) => k in f(k = 0.5)");
        assert_clean("let f = fn(k = [1, 2]) => 1 in f(k = [time])");
        assert_clean("let f = fn(k = sqrt) => 1 in f(k = cos)");
        // The omitted conjunct is the default's own type, which is what keeps
        // this sound: a call that leaves the argument out gets the result
        // the default produces, not a fresh unknown.
        assert_errors(
            "let f = fn(k = [1, 2]) => nth(0, k) in f() \\ 1",
            &["expected seq, found int"],
        );
        assert_clean("let f = fn(k = true) => k in f()");
        // A body that cannot use the argument still rejects it.
        assert_errors(
            "let f = fn(k = 1) => nth(k, [1, 2]) in f(k = time)",
            &["expected int, found waveform"],
        );
    }

    #[test]
    fn an_arity_mismatch_reports_once() {
        // Pairing arguments against parameters they were never going to line
        // up with reports a mismatch per position the shift moved. The arity
        // is the one thing wrong, so it is the one thing said.
        assert_errors(
            "let f = fn(a, b) => nth(a, b) in f(\"s\")",
            &["missing parameter of type [?a]"],
        );
        assert_errors(
            "let f = fn(a, b) => nth(a, b) in f(\"s\", [1, 2], true)",
            &["extra positional parameter"],
        );
        // Which names a call passes does not depend on how many positions it
        // got right, so a bad name is still its own error.
        assert_errors(
            "sine(440, y = 1)",
            &[
                "missing parameter of type waveform",
                "no named parameter \"y\"",
            ],
        );
        // A call of the right arity is judged as before.
        assert_errors(
            "let f = fn(a, b) => nth(a, b) in f(\"s\", [1, 2])",
            &["expected int, found string"],
        );
        assert_clean("let f = fn(a, b) => nth(a, b) in f(1, [1, 2])");
    }

    #[test]
    fn a_use_without_a_path_is_reported() {
        // The parser only builds an empty path when the path failed to
        // parse, so this is unreachable from source — but the checker must
        // not accept a binding evaluation refuses.
        let bindings: Vec<SourceBinding<u32, u32>> = vec![Binding::Use(Vec::new()).into()];
        let expr = parse_program::<u32, _>("0", 9999).unwrap();
        let resolve =
            |path: &[String]| Err(Error::eval_here(format!("no module {}", path.join("."))));
        let errors = check_program(resolve, &bindings, &expr, None);
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("requires a module path")),
            "expected a report, got {:?}",
            errors.iter().map(|e| e.message()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn modules_sharing_a_start_address_are_told_apart() {
        // A prefix of a module's bindings starts where the module does, so a
        // key of the pointer alone would make the two the same module: the
        // first one built would answer for both, and the second's exports
        // would never be seen.
        let (whole, _) = parse_module::<u32, _>("x = 1;\ny = 2;\n", 0u32).unwrap();
        let prefix = &whole[..1];
        assert_eq!(
            whole.as_ptr() as usize,
            prefix.as_ptr() as usize,
            "the prefix should share the module's start address"
        );
        let resolve = |path: &[String]| match path.join(".").as_str() {
            "whole" => Ok(&whole[..]),
            "prefix" => Ok(prefix),
            other => Err(Error::eval_here(format!("no module {}", other))),
        };
        // `whole` exports both names; `prefix` only the first. Opening both
        // must not let one stand in for the other.
        let (bindings, _) = parse_module::<u32, _>("use whole;\nuse prefix;\n", 9998u32).unwrap();
        let expr = parse_program::<u32, _>("(whole.y, prefix.x)", 9999).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert!(
            errors.is_empty(),
            "each module should keep its own exports, got {:?}",
            errors.iter().map(|e| e.message()).collect::<Vec<_>>()
        );
        // And the name the prefix does not export is still absent from it.
        let expr = parse_program::<u32, _>("prefix.y", 9999).unwrap();
        let errors = check_program(resolve, &bindings, &expr, None);
        assert!(
            errors.iter().any(|error| error.message().contains("y")),
            "expected the prefix to lack 'y', got {:?}",
            errors.iter().map(|e| e.message()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_cyclic_open_is_reported_not_recursed() {
        // Two modules that open each other. The memo entry goes in before
        // the recursion, so the second visit finds it and stops; before that
        // this overflowed the stack.
        let (a, _) = parse_module::<u32, _>("open b;\nx = 1;\n", 0u32).unwrap();
        let (b, _) = parse_module::<u32, _>("open a;\ny = 2;\n", 1u32).unwrap();
        let resolve = |path: &[String]| match path.join(".").as_str() {
            "a" => Ok(&a[..]),
            "b" => Ok(&b[..]),
            other => Err(Error::eval_here(format!("no module {}", other))),
        };
        let expr = parse_program::<u32, _>("0", 9999).unwrap();
        let errors = check_program(resolve, &a, &expr, None);
        let messages: Vec<&str> = errors.iter().map(|error| error.message()).collect();
        assert!(
            messages.iter().any(|m| m.contains("is opened from itself")),
            "expected a cycle report, got {:?}",
            messages
        );
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
                .ok_or_else(|| Error::types_here(format!("no module {}", key)))
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
