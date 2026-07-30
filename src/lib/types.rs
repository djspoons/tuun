//! The tuun type language used by the static checker.
//!
//! Types never appear in tuun's concrete syntax; they exist only inside the
//! checker (see [`crate::infer`]) and in the text of its warnings.
//!
//! The shape follows Xie and Oliveira, "Let Arguments Go First" (ESOP 2018),
//! §3.1: types `A, B ::= a | A -> B | ∀a.A | Int`, extended with tuun's base,
//! tuple, list, and module types. Meta (unification) variables `α̂` come from
//! the algorithmic system (§3.5 and Appendix E.1), which replaces the
//! declarative system's guessed monotypes with solvable unknowns.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;

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
    /// A number: a constant waveform (`Waveform::Const`) and the value form of
    /// a number literal. Subset of `Waveform`.
    Float,
    Bool,
    String,
    Waveform,
    Seq,
    /// An n-ary function applied all at once; `named` parameters are
    /// optional-with-default at call sites.
    Function {
        positional: Vec<Type>,
        named: Vec<(String, Type)>,
        result: Box<Type>,
    },
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
            Type::Var(_)
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => self.clone(),
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
            Type::Var(_)
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => {}
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
            Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => false,
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().any(|t| t.contains_var(var, subst))
                    || named.iter().any(|(_, t)| t.contains_var(var, subst))
                    || result.contains_var(var, subst)
            }
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
            Type::Meta(_)
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => self.clone(),
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
            Type::Var(_)
            | Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => self.clone(),
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
            Type::Float
            | Type::Bool
            | Type::String
            | Type::Waveform
            | Type::Seq
            | Type::Dynamic => {}
            Type::Function {
                positional,
                named,
                result,
            } => {
                positional.iter().for_each(|t| self.visit(t));
                named.iter().for_each(|(_, t)| self.visit(t));
                self.visit(result);
            }
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
        Type::Float => write!(f, "float"),
        Type::Bool => write!(f, "bool"),
        Type::String => write!(f, "string"),
        Type::Waveform => write!(f, "waveform"),
        Type::Seq => write!(f, "seq"),
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
    fn display_base_and_compound_types() {
        assert_eq!(Type::Float.to_string(), "float");
        assert_eq!(
            Type::function(vec![Type::Waveform, Type::Waveform], Type::Waveform).to_string(),
            "(waveform, waveform) -> waveform"
        );
        assert_eq!(
            Type::List(Box::new(Type::Tuple(vec![Type::Float, Type::String]))).to_string(),
            "[(float, string)]"
        );
        assert_eq!(
            Type::Function {
                positional: vec![Type::Float],
                named: vec![("y".to_string(), Type::Float)],
                result: Box::new(Type::Float),
            }
            .to_string(),
            "(float, y: float) -> float"
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
        subst.insert(1, Type::Float);
        assert_eq!(
            Type::List(Box::new(Type::Meta(0))).apply(&subst),
            Type::List(Box::new(Type::Float))
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
    fn contains_var_respects_shadowing() {
        let subst = HashMap::new();
        let ty = Type::Forall(vec![0], Box::new(Type::Var(0)));
        assert!(!ty.contains_var(0, &subst));
        let ty = Type::Forall(vec![1], Box::new(Type::Var(0)));
        assert!(ty.contains_var(0, &subst));
    }
}
