//! The tuun type language: a predicative polymorphic lambda calculus extended
//! with:
//!
//!  * Compound types: lists, tuples, and modules (a limited form of record)
//!  * Multiple parameter functions with optional named parameters
//!  * Intersection types (limited to intersections of functions)
//!  * Base types: boolean, string, and numeric
//!
//! The numeric type describes values which are subject to arithmetic operations
//! (e.g., +, *, pow). These include waveforms (both sequenced and unsequenced),
//! floating point numbers, and integers (which are represented as constant
//! waveforms). A numeric type may be refined by a [`Sort`] which distinguishes
//! these cases. However, unseq numeric values are uniformly represented:
//! floating point numbers and integers are represented as constant waveforms.
//!
//! Types also include meta variables that are used during type inference (see
//! [`crate::infer`]) and a "dynamic" type, which is used to recover from a type
//! error and for certain built-in functions that lack a precise signature
//! (e.g., `debug`).
//!
//! Tuun types currently do not appear in the concrete syntax.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::rc::Rc;

/// A numeric sort: a point of the refinement lattice over tuun's numeric
/// values.
///
/// This a union of the four disjoint atoms, ordered by inclusion (join =
/// union). "Sort" follows the refinement-types literature, where every lattice
/// point, atomic or not, is a sort.
///
/// The atoms:
///  * `I` — an integer-valued constant waveform
///  * `NonInt` — any other constant waveform
///  * `W` — a non-constant waveform without a sequencing offset
///  * `S` — a seq (waveform with sequencing offset)
///
/// Atoms appear in the following lattice:
/// ```text
///              ⊤  "some numeric"  (the join; inference's unknown)
///             /  \
///      waveform   Seq             no edge between them —
///       (unseq)                   unseq() : ({S}) → ({W}) is the
///          |                      only crossing
///        Float                    (the constant waveforms; = Int ∨ NonInt)
///         /  \
///       Int   NonInt              (disjoint: integer-valued or not)
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Sort(u8);

impl Sort {
    const I: u8 = 1;
    const NON_INT: u8 = 2;
    const W: u8 = 4;
    const S: u8 = 8;

    /// The empty sort — no atoms (the bottom; not a valid ground
    /// refinement, but the starting lower bound of a refinement variable).
    pub const NONE: Sort = Sort(0);
    /// Integer-valued constants.
    pub const INT: Sort = Sort(Sort::I);
    /// Non-integer constants.
    pub const NON_INT_ONLY: Sort = Sort(Sort::NON_INT);
    /// All constants: `Int ∨ NonInt`.
    pub const FLOAT: Sort = Sort(Sort::I | Sort::NON_INT);
    /// Non-constant unseq waveforms.
    pub const NON_CONST_WAVE: Sort = Sort(Sort::W);
    /// Sequence-able waveforms.
    pub const SEQ: Sort = Sort(Sort::S);
    /// The waveforms: everything usable as one, constants included.
    pub const WAVE: Sort = Sort(Sort::I | Sort::NON_INT | Sort::W);
    /// The top: any stream of numeric values.
    pub const TOP: Sort = Sort(Sort::I | Sort::NON_INT | Sort::W | Sort::S);

    pub fn union(self, other: Sort) -> Sort {
        Sort(self.0 | other.0)
    }

    pub fn intersect(self, other: Sort) -> Sort {
        Sort(self.0 & other.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether every atom in `self` is also in `other`.
    pub fn is_subset(self, other: Sort) -> bool {
        self.0 & !other.0 == 0
    }

    /// Returns the atoms of this sort, each as a singleton.
    pub fn atoms(self) -> impl Iterator<Item = Sort> {
        [
            Sort::INT,
            Sort::NON_INT_ONLY,
            Sort::NON_CONST_WAVE,
            Sort::SEQ,
        ]
        .into_iter()
        .filter(move |atom| !self.intersect(*atom).is_empty())
    }
}

impl Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Sort::TOP {
            return write!(f, "numeric");
        }
        // Compose atom names, collapsing unions to the name of the class
        // they fill: the whole waveform class (constants included) is just
        // "waveform", and both constant atoms together are "float".
        // Partial atoms render as their class too — `NonInt` alone as
        // "float", `NON_CONST_WAVE` alone as "waveform" — because the
        // finer split only matters against a narrower contract, where the
        // caret shows the offending expression.
        let mut parts: Vec<&str> = Vec::new();
        if Sort::WAVE.is_subset(*self) {
            parts.push("waveform");
        } else {
            if Sort::FLOAT.is_subset(*self) {
                parts.push("float");
            } else if !self.intersect(Sort::INT).is_empty() {
                parts.push("int");
            } else if !self.intersect(Sort::NON_INT_ONLY).is_empty() {
                parts.push("float");
            }
            if !self.intersect(Sort::NON_CONST_WAVE).is_empty() {
                parts.push("waveform");
            }
        }
        if !self.intersect(Sort::SEQ).is_empty() {
            parts.push("seq");
        }
        if parts.is_empty() {
            return write!(f, "nothing");
        }
        write!(f, "{}", parts.join(" or "))
    }
}

/// The refinement carried by a numeric type: a known sort or a refinement
/// variable solved by the checker (bounds live in the checker's state, not
/// here).
#[derive(Clone, Debug, PartialEq)]
pub enum Refinement {
    Ground(Sort),
    Var(u32),
}

/// A tuun type.
///
/// `Meta` variables are unification variables solved during inference; `Var`s
/// are rigid variables bound by `Forall`. The system is predicative (paper
/// §2.4): metas are only ever solved to types without quantifiers.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// A rigid type variable bound by an enclosing `Forall` (the paper's
    /// `a`, `b`), also used as a skolem during subtyping (rule AS-ForallR).
    Var(u32),
    /// A unification variable (the paper's `α̂`, `β̂`; Appendix E.1), solved
    /// via the checker's substitution `S`.
    Meta(u32),
    /// A numeric value (a stream of floats), refined by a sort.
    Numeric(Refinement),
    Bool,
    String,
    /// An n-ary function applied all at once; `named` parameters are
    /// optional-with-default at call sites.
    Function {
        positional: Rc<[Type]>,
        named: Rc<[(String, Type)]>,
        result: Rc<Type>,
    },
    /// An intersection of types.
    ///
    /// Always a set of arrows refining a common function type, for example, the
    /// principal type of an overloaded built-in function.
    And(Rc<[Type]>),
    Tuple(Vec<Type>),
    List(Box<Type>),
    /// The type of a module value (from `use`); one entry per exported
    /// binding.
    Module(Vec<(String, Type)>),
    /// A polymorphic type `∀ā.A` quantifying the listed `Var` ids. Always
    /// prenex: produced only by generalization (rule AT-Gen, Fig. 16).
    Forall(Vec<u32>, Box<Type>),
    /// The escape hatch: compatible with every type in every position.
    ///
    /// Used for built-ins without a precise signature and to recover after a
    /// reported error without cascading follow-on errors.
    Dynamic,
}

impl Type {
    /// Builds a function type from positional parameter types and a result.
    pub fn function(positional: Vec<Type>, result: Type) -> Type {
        Type::Function {
            positional: positional.into(),
            named: Rc::from(Vec::new()),
            result: Rc::new(result),
        }
    }

    /// Builds the intersection of `types`, which is the type itself where
    /// there is only one of them.
    ///
    /// A one-conjunct intersection selects and displays exactly as its conjunct
    /// does, so building one only makes the type harder to read.
    pub fn intersection(mut types: Vec<Type>) -> Type {
        if types.len() == 1 {
            return types.remove(0);
        }
        Type::And(types.into())
    }

    /// Builds the numeric type with exactly the given sort.
    pub fn ground(sort: Sort) -> Type {
        Type::Numeric(Refinement::Ground(sort))
    }

    /// The integer-valued constants.
    pub fn int() -> Type {
        Type::ground(Sort::INT)
    }

    /// The constants (integer-valued or not).
    pub fn float() -> Type {
        Type::ground(Sort::FLOAT)
    }

    /// The non-constant unseq waveforms.
    pub fn non_const_wave() -> Type {
        Type::ground(Sort::NON_CONST_WAVE)
    }

    /// The seqs.
    pub fn seq() -> Type {
        Type::ground(Sort::SEQ)
    }

    /// The waveforms: everything usable as one, constants included.
    pub fn waveform() -> Type {
        Type::ground(Sort::WAVE)
    }

    /// Any numeric value.
    pub fn numeric() -> Type {
        Type::ground(Sort::TOP)
    }

    /// Whether any refinement variable appears in this type.
    ///
    /// Syntactic, like the freezing it guards: freezing rewrites exactly these
    /// nodes and leaves a meta alone, so a type without one is already what
    /// freezing would return.
    pub fn has_refinement_var(&self) -> bool {
        match self {
            Type::Numeric(Refinement::Var(_)) => true,
            Type::Meta(_)
            | Type::Var(_)
            | Type::Numeric(_)
            | Type::Bool
            | Type::String
            | Type::Dynamic => false,
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().any(Type::has_refinement_var)
                    || named.iter().any(|(_, ty)| ty.has_refinement_var())
                    || result.has_refinement_var()
            }
            Type::And(types) => types.iter().any(Type::has_refinement_var),
            Type::Tuple(types) => types.iter().any(Type::has_refinement_var),
            Type::List(item) => item.has_refinement_var(),
            Type::Module(entries) => entries.iter().any(|(_, ty)| ty.has_refinement_var()),
            Type::Forall(_, body) => body.has_refinement_var(),
        }
    }

    /// Whether this type holds no meta and no refinement variable.
    ///
    /// Syntactic, and deliberately not read through a substitution: a meta that
    /// is solved now can be unsolved again by a rollback, so only a type that
    /// never mentions one is settled for good.
    pub fn settled(&self) -> bool {
        match self {
            Type::Meta(_) | Type::Numeric(Refinement::Var(_)) => false,
            Type::Var(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => true,
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().all(Type::settled)
                    && named.iter().all(|(_, ty)| ty.settled())
                    && result.settled()
            }
            Type::And(types) => types.iter().all(Type::settled),
            Type::Tuple(types) => types.iter().all(Type::settled),
            Type::List(item) => item.settled(),
            Type::Module(entries) => entries.iter().all(|(_, ty)| ty.settled()),
            Type::Forall(_, body) => body.settled(),
        }
    }

    /// Returns this type with `subst` applied deeply: every solved `Meta` is
    /// replaced by its (recursively applied) solution.
    ///
    /// The paper writes this `S A` (Appendix E.1).
    pub fn apply(&self, subst: &HashMap<u32, Type>) -> Type {
        match self {
            Type::Meta(id) => match subst.get(id) {
                Some(solution) => solution.apply(subst),
                None => self.clone(),
            },
            Type::Var(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {
                self.clone()
            }
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional.iter().map(|t| t.apply(subst)).collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), t.apply(subst)))
                    .collect(),
                result: Rc::new(result.apply(subst)),
            },
            Type::And(conjuncts) => Type::And(conjuncts.iter().map(|t| t.apply(subst)).collect()),
            Type::Tuple(items) => Type::Tuple(items.iter().map(|t| t.apply(subst)).collect()),
            Type::List(item) => Type::List(Box::new(item.apply(subst))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), t.apply(subst)))
                    .collect(),
            ),
            Type::Forall(vars, body) => Type::Forall(vars.clone(), Box::new(body.apply(subst))),
        }
    }

    /// Collects the unsolved `Meta` ids of this type into `acc`, looking
    /// through `subst`, in first-appearance order and without duplicates.
    ///
    /// Computes the paper's `ftv(S A)` restricted to meta variables, as used
    /// by generalization (rule AT-Gen, Fig. 16).
    pub fn free_metas(&self, subst: &HashMap<u32, Type>, acc: &mut Vec<u32>) {
        match self {
            Type::Meta(id) => match subst.get(id) {
                Some(solution) => solution.free_metas(subst, acc),
                None => {
                    if !acc.contains(id) {
                        acc.push(*id);
                    }
                }
            },
            Type::Var(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {}
            Type::Function {
                positional,
                named,
                result,
            } => {
                for t in positional.iter() {
                    t.free_metas(subst, acc);
                }
                for (_, t) in named.iter() {
                    t.free_metas(subst, acc);
                }
                result.free_metas(subst, acc);
            }
            Type::And(conjuncts) => {
                for t in conjuncts.iter() {
                    t.free_metas(subst, acc);
                }
            }
            Type::Tuple(items) => {
                for t in items {
                    t.free_metas(subst, acc);
                }
            }
            Type::List(item) => item.free_metas(subst, acc),
            Type::Module(entries) => {
                for (_, t) in entries {
                    t.free_metas(subst, acc);
                }
            }
            Type::Forall(_, body) => body.free_metas(subst, acc),
        }
    }

    /// Collects the refinement-variable ids of this type into `acc`, looking
    /// through `subst` for solved metas, without duplicates.
    ///
    /// The refinement analog of [`Type::free_metas`], used by generalization to
    /// leave context-reachable refinement variables unfrozen.
    pub fn free_refinements(&self, subst: &HashMap<u32, Type>, acc: &mut Vec<u32>) {
        match self {
            Type::Numeric(Refinement::Var(id)) => {
                if !acc.contains(id) {
                    acc.push(*id);
                }
            }
            Type::Meta(id) => {
                if let Some(solution) = subst.get(id) {
                    solution.free_refinements(subst, acc);
                }
            }
            Type::Var(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {}
            Type::Function {
                positional,
                named,
                result,
            } => {
                for t in positional.iter() {
                    t.free_refinements(subst, acc);
                }
                for (_, t) in named.iter() {
                    t.free_refinements(subst, acc);
                }
                result.free_refinements(subst, acc);
            }
            Type::And(conjuncts) => {
                for t in conjuncts.iter() {
                    t.free_refinements(subst, acc);
                }
            }
            Type::Tuple(items) => {
                for t in items {
                    t.free_refinements(subst, acc);
                }
            }
            Type::List(item) => item.free_refinements(subst, acc),
            Type::Module(entries) => {
                for (_, t) in entries {
                    t.free_refinements(subst, acc);
                }
            }
            Type::Forall(_, body) => body.free_refinements(subst, acc),
        }
    }

    /// Returns whether the rigid variable `var` occurs in this type, looking
    /// through `subst`.
    ///
    /// Implements the freshness side conditions `b ∉ ftv(...)` of rule
    /// AS-ForallR (Fig. 15): a skolem escaping into the left-hand type means
    /// that type is not actually as polymorphic as required.
    pub fn contains_var(&self, var: u32, subst: &HashMap<u32, Type>) -> bool {
        match self {
            Type::Var(id) => *id == var,
            Type::Meta(id) => match subst.get(id) {
                Some(solution) => solution.contains_var(var, subst),
                None => false,
            },
            Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => false,
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().any(|t| t.contains_var(var, subst))
                    || named.iter().any(|(_, t)| t.contains_var(var, subst))
                    || result.contains_var(var, subst)
            }
            Type::And(conjuncts) => conjuncts.iter().any(|t| t.contains_var(var, subst)),
            Type::Tuple(items) => items.iter().any(|t| t.contains_var(var, subst)),
            Type::List(item) => item.contains_var(var, subst),
            Type::Module(entries) => entries.iter().any(|(_, t)| t.contains_var(var, subst)),
            Type::Forall(vars, body) => !vars.contains(&var) && body.contains_var(var, subst),
        }
    }

    /// Returns this type with each `Var` in `mapping` replaced by the mapped
    /// type.
    ///
    /// The paper writes this `A[a ↦ τ]`; it instantiates `Forall` bodies
    /// (rules S-ForallL, AS-ForallL, AS-ForallL2) and freshens quantified
    /// variables when skolemizing (rule AS-ForallR).
    pub fn substitute_vars(&self, mapping: &HashMap<u32, Type>) -> Type {
        match self {
            Type::Var(id) => match mapping.get(id) {
                Some(replacement) => replacement.clone(),
                None => self.clone(),
            },
            Type::Meta(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {
                self.clone()
            }
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional
                    .iter()
                    .map(|t| t.substitute_vars(mapping))
                    .collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute_vars(mapping)))
                    .collect(),
                result: Rc::new(result.substitute_vars(mapping)),
            },
            Type::And(conjuncts) => Type::And(
                conjuncts
                    .iter()
                    .map(|t| t.substitute_vars(mapping))
                    .collect(),
            ),
            Type::Tuple(items) => {
                Type::Tuple(items.iter().map(|t| t.substitute_vars(mapping)).collect())
            }
            Type::List(item) => Type::List(Box::new(item.substitute_vars(mapping))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute_vars(mapping)))
                    .collect(),
            ),
            Type::Forall(vars, body) => {
                // Quantified vars shadow the mapping inside the body.
                let mut inner = mapping.clone();
                for var in vars {
                    inner.remove(var);
                }
                Type::Forall(vars.clone(), Box::new(body.substitute_vars(&inner)))
            }
        }
    }

    /// Returns this type with each meta in `mapping` replaced by the mapped
    /// type.
    ///
    /// Unlike [`Type::apply`], this rewrites *unsolved* metas; generalization
    /// uses it to close them over as fresh rigid variables — the paper's
    /// `(S0 A)[ᾱ ↦ b̄]` in rule AT-Gen (Fig. 16).
    pub fn substitute_metas(&self, mapping: &HashMap<u32, Type>) -> Type {
        match self {
            Type::Meta(id) => match mapping.get(id) {
                Some(replacement) => replacement.clone(),
                None => self.clone(),
            },
            Type::Var(_) | Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {
                self.clone()
            }
            Type::Function {
                positional,
                named,
                result,
            } => Type::Function {
                positional: positional
                    .iter()
                    .map(|t| t.substitute_metas(mapping))
                    .collect(),
                named: named
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute_metas(mapping)))
                    .collect(),
                result: Rc::new(result.substitute_metas(mapping)),
            },
            Type::And(conjuncts) => Type::And(
                conjuncts
                    .iter()
                    .map(|t| t.substitute_metas(mapping))
                    .collect(),
            ),
            Type::Tuple(items) => {
                Type::Tuple(items.iter().map(|t| t.substitute_metas(mapping)).collect())
            }
            Type::List(item) => Type::List(Box::new(item.substitute_metas(mapping))),
            Type::Module(entries) => Type::Module(
                entries
                    .iter()
                    .map(|(n, t)| (n.clone(), t.substitute_metas(mapping)))
                    .collect(),
            ),
            Type::Forall(vars, body) => {
                Type::Forall(vars.clone(), Box::new(body.substitute_metas(mapping)))
            }
        }
    }
}

/// Assigns display names (`'a`, `'b`, ..., `'a1`, ...) to variable ids in
/// first-appearance order, so rendered types read the same regardless of the
/// raw ids inference happened to allocate.
struct Names {
    vars: Vec<u32>,
    metas: Vec<u32>,
}

impl Names {
    fn collect(ty: &Type) -> Names {
        let mut names = Names {
            vars: Vec::new(),
            metas: Vec::new(),
        };
        names.visit(ty);
        names
    }

    fn visit(&mut self, ty: &Type) {
        match ty {
            Type::Var(id) => {
                if !self.vars.contains(id) {
                    self.vars.push(*id);
                }
            }
            Type::Meta(id) => {
                if !self.metas.contains(id) {
                    self.metas.push(*id);
                }
            }
            Type::Numeric(_) | Type::Bool | Type::String | Type::Dynamic => {}
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().for_each(|t| self.visit(t));
                named.iter().for_each(|(_, t)| self.visit(t));
                self.visit(result);
            }
            Type::And(conjuncts) => conjuncts.iter().for_each(|t| self.visit(t)),
            Type::Tuple(items) => items.iter().for_each(|t| self.visit(t)),
            Type::List(item) => self.visit(item),
            Type::Module(entries) => entries.iter().for_each(|(_, t)| self.visit(t)),
            Type::Forall(_, body) => self.visit(body),
        }
    }

    fn letters(index: usize) -> std::string::String {
        let letter = (b'a' + (index % 26) as u8) as char;
        let round = index / 26;
        if round == 0 {
            letter.to_string()
        } else {
            format!("{}{}", letter, round)
        }
    }

    fn var(&self, id: u32) -> std::string::String {
        let index = self.vars.iter().position(|v| *v == id).unwrap_or(0);
        format!("'{}", Names::letters(index))
    }

    fn meta(&self, id: u32) -> std::string::String {
        let index = self.metas.iter().position(|v| *v == id).unwrap_or(0);
        format!("?{}", Names::letters(index))
    }
}

/// Renders `ty`, with `nested` set where it is written as part of another
/// type.
///
/// An intersection carries no delimiters of its own, so one nested inside
/// another type needs parentheses to say where it ends: without them
/// `(int) -> (int) -> int ∧ (float) -> float` does not distinguish a result
/// that is an intersection from a conjunct of the intersection around it.
/// Brackets and a parameter list already delimit what they contain, so a
/// list element does not need them.
fn fmt_type(ty: &Type, names: &Names, f: &mut fmt::Formatter<'_>, nested: bool) -> fmt::Result {
    match ty {
        Type::Var(id) => write!(f, "{}", names.var(*id)),
        Type::Meta(id) => write!(f, "{}", names.meta(*id)),
        Type::Numeric(Refinement::Ground(sort)) => write!(f, "{}", sort),
        // Unresolved refinement variables read as "unknown numeric"; the
        // checker resolves known bounds before display.
        Type::Numeric(Refinement::Var(_)) => write!(f, "number"),
        Type::Bool => write!(f, "bool"),
        Type::String => write!(f, "string"),
        Type::Dynamic => write!(f, "dynamic"),
        Type::Function {
            positional,
            named,
            result,
        } => {
            write!(f, "(")?;
            let mut first = true;
            for t in positional.iter() {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                fmt_type(t, names, f, true)?;
            }
            for (name, t) in named.iter() {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                write!(f, "{}: ", name)?;
                fmt_type(t, names, f, true)?;
            }
            write!(f, ") -> ")?;
            fmt_type(result, names, f, true)
        }
        Type::And(conjuncts) => {
            if nested {
                write!(f, "(")?;
            }
            for (i, t) in conjuncts.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∧ ")?;
                }
                fmt_type(t, names, f, true)?;
            }
            if nested {
                write!(f, ")")?;
            }
            Ok(())
        }
        Type::Tuple(items) => {
            write!(f, "(")?;
            for (i, t) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_type(t, names, f, true)?;
            }
            write!(f, ")")?;
            Ok(())
        }
        Type::List(item) => {
            write!(f, "[")?;
            fmt_type(item, names, f, false)?;
            write!(f, "]")
        }
        Type::Module(entries) => {
            write!(f, "module {{")?;
            for (i, (name, _)) in entries.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", name)?;
            }
            write!(f, "}}")
        }
        // Quantifiers are left implicit: the quantified vars simply render by
        // name, HM-style.
        Type::Forall(_, body) => fmt_type(body, names, f, nested),
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_type(self, &Names::collect(self), f, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_order_by_inclusion() {
        assert!(Sort::INT.is_subset(Sort::FLOAT));
        assert!(Sort::FLOAT.is_subset(Sort::WAVE));
        assert!(!Sort::SEQ.is_subset(Sort::WAVE));
        assert!(Sort::SEQ.intersect(Sort::NON_CONST_WAVE).is_empty());
        assert_eq!(Sort::FLOAT.union(Sort::NON_CONST_WAVE), Sort::WAVE);
    }

    #[test]
    fn display_sorts() {
        assert_eq!(Sort::INT.to_string(), "int");
        assert_eq!(Sort::NON_INT_ONLY.to_string(), "float");
        assert_eq!(Sort::FLOAT.to_string(), "float");
        assert_eq!(Sort::NON_CONST_WAVE.to_string(), "waveform");
        assert_eq!(Sort::SEQ.to_string(), "seq");
        assert_eq!(Sort::WAVE.to_string(), "waveform");
        assert_eq!(Sort::TOP.to_string(), "numeric");
        assert_eq!(
            Sort::INT.union(Sort::NON_CONST_WAVE).to_string(),
            "int or waveform"
        );
    }

    #[test]
    fn display_base_and_compound_types() {
        assert_eq!(Type::float().to_string(), "float");
        assert_eq!(
            Type::function(
                vec![Type::non_const_wave(), Type::non_const_wave()],
                Type::non_const_wave()
            )
            .to_string(),
            "(waveform, waveform) -> waveform"
        );
        assert_eq!(
            Type::List(Box::new(Type::Tuple(vec![Type::float(), Type::String]))).to_string(),
            "[(float, string)]"
        );
        assert_eq!(
            Type::Function {
                positional: vec![Type::float()].into(),
                named: vec![("y".to_string(), Type::float())].into(),
                result: Rc::new(Type::float()),
            }
            .to_string(),
            "(float, y: float) -> float"
        );
        assert_eq!(
            Type::intersection(vec![
                Type::function(vec![Type::int()], Type::int()),
                Type::function(vec![Type::float()], Type::float()),
            ])
            .to_string(),
            "(int) -> int ∧ (float) -> float"
        );
    }

    #[test]
    fn nested_intersections_are_parenthesised() {
        let inner = Type::intersection(vec![
            Type::function(vec![Type::int()], Type::int()),
            Type::function(vec![Type::float()], Type::float()),
        ]);
        // At the top there is nothing an intersection could be confused
        // with, so it needs no parentheses.
        assert_eq!(inner.to_string(), "(int) -> int ∧ (float) -> float");
        // In a result position it does: without them the inner conjuncts
        // read as conjuncts of the intersection around them, which is how a
        // curried tabulated definition prints.
        let curried = Type::intersection(vec![
            Type::function(vec![Type::int()], inner.clone()),
            Type::function(vec![Type::waveform()], Type::waveform()),
        ]);
        assert_eq!(
            curried.to_string(),
            "(int) -> ((int) -> int ∧ (float) -> float) ∧ (waveform) -> waveform"
        );
        // A tuple's commas have the same problem, so its components are
        // parenthesised too — inside the tuple's own parentheses.
        assert_eq!(
            Type::Tuple(vec![inner.clone(), Type::String]).to_string(),
            "(((int) -> int ∧ (float) -> float), string)"
        );
        assert_eq!(
            Type::List(Box::new(inner)).to_string(),
            "[(int) -> int ∧ (float) -> float]"
        );
    }

    #[test]
    fn display_names_vars_in_appearance_order() {
        // ∀-quantified vars render as 'a, 'b, ... regardless of raw ids.
        let ty = Type::Forall(
            vec![7, 3],
            Box::new(Type::function(
                vec![Type::Var(7), Type::Var(3)],
                Type::Var(7),
            )),
        );
        assert_eq!(ty.to_string(), "('a, 'b) -> 'a");
    }

    #[test]
    fn apply_follows_solution_chains() {
        let mut subst = HashMap::new();
        subst.insert(0, Type::Meta(1));
        subst.insert(1, Type::float());
        assert_eq!(
            Type::List(Box::new(Type::Meta(0))).apply(&subst),
            Type::List(Box::new(Type::float()))
        );
    }

    #[test]
    fn free_metas_looks_through_substitution() {
        let mut subst = HashMap::new();
        subst.insert(0, Type::List(Box::new(Type::Meta(2))));
        let ty = Type::function(vec![Type::Meta(0), Type::Meta(1)], Type::Meta(1));
        let mut metas = Vec::new();
        ty.free_metas(&subst, &mut metas);
        assert_eq!(metas, vec![2, 1]);
    }

    #[test]
    fn free_refinements_looks_through_substitution() {
        let mut subst = HashMap::new();
        subst.insert(0, Type::Numeric(Refinement::Var(7)));
        let ty = Type::function(vec![Type::Meta(0)], Type::Numeric(Refinement::Var(9)));
        let mut refinements = Vec::new();
        ty.free_refinements(&subst, &mut refinements);
        assert_eq!(refinements, vec![7, 9]);
    }

    #[test]
    fn contains_var_respects_shadowing() {
        let subst = HashMap::new();
        let ty = Type::Forall(vec![0], Box::new(Type::Var(0)));
        assert!(!ty.contains_var(0, &subst));
        let ty = Type::Forall(vec![1], Box::new(Type::Var(0)));
        assert!(ty.contains_var(0, &subst));
    }
}
