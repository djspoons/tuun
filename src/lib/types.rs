//! The tuun type language used by the static checker.
//!
//! Types never appear in tuun's concrete syntax; they exist only inside the
//! checker (see [`crate::infer`]) and in the text of its warnings.
//!
//! The structure follows Xie and Oliveira, "Let Arguments Go First" (ESOP
//! 2018), §3.1: types `A, B ::= a | A -> B | ∀a.A | Int`, extended with tuun's
//! base, tuple, list, and module types. Meta (unification) variables `α̂` come
//! from the algorithmic system (§3.5 and Appendix E.1), which replaces the
//! declarative system's guessed monotypes with solvable unknowns.
//!
//! The Numeric type represents waveforms and related values: streams of
//! floating point numbers. Literals (like `3`) are interpreted as constant
//! waveforms. Numeric is refined by a [`Sort`], a union of four disjoint atoms,
//! and overloaded built-in functions carry intersections of arrows
//! ([`Type::And`]) in the style of Freeman and Pfenning, "Refinement Types for
//! ML" (PLDI 1991).

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;

/// A numeric sort: a point of the checker's refinement lattice over tuun's
/// numeric values, a union of the four disjoint atoms, ordered by inclusion
/// (join = union). "Sort" follows the refinement-types literature, where every
/// lattice point, atomic or not, is a sort.
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
    pub const WAVE: Sort = Sort(Sort::W);
    /// Sequence-able waveforms.
    pub const SEQ: Sort = Sort(Sort::S);
    /// Everything a waveform position accepts: constants and unseq waveforms.
    pub const FLOAT_OR_WAVE: Sort = Sort(Sort::I | Sort::NON_INT | Sort::W);
    /// Waveform or seq.
    pub const WAVE_OR_SEQ: Sort = Sort(Sort::W | Sort::S);
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
        [Sort::INT, Sort::NON_INT_ONLY, Sort::WAVE, Sort::SEQ]
            .into_iter()
            .filter(move |atom| !self.intersect(*atom).is_empty())
    }
}

impl Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Sort::TOP {
            return write!(f, "number");
        }
        // Compose atom names, collapsing the two constant atoms to "float"
        // when both are present; `NonInt` alone also renders as "float" (the
        // int/non-int distinction only matters against an int contract,
        // where the caret shows the offending literal).
        let mut parts: Vec<&str> = Vec::new();
        if Sort::FLOAT.is_subset(*self) {
            parts.push("float");
        } else if !self.intersect(Sort::INT).is_empty() {
            parts.push("int");
        } else if !self.intersect(Sort::NON_INT_ONLY).is_empty() {
            parts.push("float");
        }
        if !self.intersect(Sort::WAVE).is_empty() {
            parts.push("waveform");
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
        positional: Vec<Type>,
        named: Vec<(String, Type)>,
        result: Box<Type>,
    },
    /// An intersection of types.
    ///
    /// Currently always a set of arrows refining a common function type, for
    /// example, the principal type of an overloaded built-in function.
    And(Vec<Type>),
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
    /// reported warning without cascading follow-on warnings.
    Dynamic,
}

impl Type {
    /// Builds a function type from positional parameter types and a result.
    pub fn function(positional: Vec<Type>, result: Type) -> Type {
        Type::Function {
            positional,
            named: Vec::new(),
            result: Box::new(result),
        }
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
    pub fn waveform() -> Type {
        Type::ground(Sort::WAVE)
    }

    /// The seqs.
    pub fn seq() -> Type {
        Type::ground(Sort::SEQ)
    }

    /// Anything a waveform position accepts: a constant or an unseq
    /// waveform.
    pub fn float_or_waveform() -> Type {
        Type::ground(Sort::FLOAT_OR_WAVE)
    }

    /// Any numeric value.
    pub fn number() -> Type {
        Type::ground(Sort::TOP)
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
                result: Box::new(result.apply(subst)),
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
                for t in positional {
                    t.free_metas(subst, acc);
                }
                for (_, t) in named {
                    t.free_metas(subst, acc);
                }
                result.free_metas(subst, acc);
            }
            Type::And(conjuncts) => {
                for t in conjuncts {
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
                for t in positional {
                    t.free_refinements(subst, acc);
                }
                for (_, t) in named {
                    t.free_refinements(subst, acc);
                }
                result.free_refinements(subst, acc);
            }
            Type::And(conjuncts) => {
                for t in conjuncts {
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
                result: Box::new(result.substitute_vars(mapping)),
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
                result: Box::new(result.substitute_metas(mapping)),
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

fn fmt_type(ty: &Type, names: &Names, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            for t in positional {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                fmt_type(t, names, f)?;
            }
            for (name, t) in named {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                write!(f, "{}: ", name)?;
                fmt_type(t, names, f)?;
            }
            write!(f, ") -> ")?;
            fmt_type(result, names, f)
        }
        Type::And(conjuncts) => {
            for (i, t) in conjuncts.iter().enumerate() {
                if i > 0 {
                    write!(f, " ∧ ")?;
                }
                fmt_type(t, names, f)?;
            }
            Ok(())
        }
        Type::Tuple(items) => {
            write!(f, "(")?;
            for (i, t) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_type(t, names, f)?;
            }
            write!(f, ")")?;
            Ok(())
        }
        Type::List(item) => {
            write!(f, "[")?;
            fmt_type(item, names, f)?;
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
        Type::Forall(_, body) => fmt_type(body, names, f),
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_type(self, &Names::collect(self), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_order_by_inclusion() {
        assert!(Sort::INT.is_subset(Sort::FLOAT));
        assert!(Sort::FLOAT.is_subset(Sort::FLOAT_OR_WAVE));
        assert!(!Sort::SEQ.is_subset(Sort::FLOAT_OR_WAVE));
        assert!(Sort::SEQ.intersect(Sort::WAVE).is_empty());
        assert_eq!(Sort::FLOAT.union(Sort::WAVE), Sort::FLOAT_OR_WAVE);
    }

    #[test]
    fn display_sorts() {
        assert_eq!(Sort::INT.to_string(), "int");
        assert_eq!(Sort::NON_INT_ONLY.to_string(), "float");
        assert_eq!(Sort::FLOAT.to_string(), "float");
        assert_eq!(Sort::WAVE.to_string(), "waveform");
        assert_eq!(Sort::SEQ.to_string(), "seq");
        assert_eq!(Sort::FLOAT_OR_WAVE.to_string(), "float or waveform");
        assert_eq!(Sort::WAVE_OR_SEQ.to_string(), "waveform or seq");
        assert_eq!(Sort::TOP.to_string(), "number");
        assert_eq!(Sort::INT.union(Sort::WAVE).to_string(), "int or waveform");
    }

    #[test]
    fn display_base_and_compound_types() {
        assert_eq!(Type::float().to_string(), "float");
        assert_eq!(
            Type::function(vec![Type::waveform(), Type::waveform()], Type::waveform()).to_string(),
            "(waveform, waveform) -> waveform"
        );
        assert_eq!(
            Type::List(Box::new(Type::Tuple(vec![Type::float(), Type::String]))).to_string(),
            "[(float, string)]"
        );
        assert_eq!(
            Type::Function {
                positional: vec![Type::float()],
                named: vec![("y".to_string(), Type::float())],
                result: Box::new(Type::float()),
            }
            .to_string(),
            "(float, y: float) -> float"
        );
        assert_eq!(
            Type::And(vec![
                Type::function(vec![Type::int()], Type::int()),
                Type::function(vec![Type::float()], Type::float()),
            ])
            .to_string(),
            "(int) -> int ∧ (float) -> float"
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
