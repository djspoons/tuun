//! Shared scaffolding for the checker's integration tests: the real prelude
//! and embedded library as a module scope, a seeded random source, and the
//! helpers that compare a checked type against an evaluated value.

// Each test crate compiles this module and uses only part of it.
#![allow(dead_code)]

use std::sync::Arc;

use tuun::builtins;
use tuun::expr::{Binding, BuiltInFn, Error, ErrorKind, Expr, Pattern, SourceBinding, SourceExpr};
use tuun::modules;
use tuun::parser::parse_module;
use tuun::types::Sort;
use tuun::waveform::Waveform;

/// One embedded library module, parsed.
pub struct LibraryModule {
    /// The module's dotted path, e.g. `env.minmax`.
    pub path: String,
    /// The source text its spans index into.
    pub text: &'static str,
    /// The span source id its bindings carry, distinct per module.
    pub source: u32,
    /// Its bindings, opening the prelude first.
    pub bindings: Vec<SourceBinding<u32, u32>>,
}

/// The modules a program checks and evaluates under: the prelude and every
/// embedded library module, as a program that opens them sees them.
pub struct Scope {
    /// The built-ins plus `tempo`, `sample_rate`, `mark`, and `debug`.
    pub prelude: Vec<SourceBinding<u32, u32>>,
    /// The embedded library, in [`modules::EMBEDDED_MODULES`] order.
    pub library: Vec<LibraryModule>,
    /// The bindings a program is checked under: it opens the prelude, then
    /// std.
    ///
    /// Opening std adds about 14 ms to every `eval::evaluate` (measured on
    /// `1 + 2`), since evaluation has no counterpart to the checker's
    /// `ModuleCache`.
    pub program: Vec<SourceBinding<u32, u32>>,
}

impl Scope {
    /// Builds the scope, giving module `i` span source `i + 1` so source 0
    /// is left for programs.
    pub fn new() -> Scope {
        let open = |path: &str| Binding::Open(vec![path.to_string()]).into();
        let mut prelude = Vec::new();
        builtins::add_bindings(&mut prelude);
        for (name, value) in [("tempo", 120.0), ("sample_rate", 44100.0)] {
            prelude.push(SourceBinding::definition(
                Pattern::Identifier(name.to_string()),
                SourceExpr::float(value),
            ));
        }
        // `mark` needs a running voice, so here it is a stub that fails.
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("mark".to_string()),
            SourceExpr::from(Expr::BuiltIn {
                name: "mark".to_string(),
                function: BuiltInFn(Arc::new(|_| Err(Error::eval_here("mark needs a voice")))),
            }),
        ));
        prelude.push(SourceBinding::definition(
            Pattern::Identifier("debug".to_string()),
            builtins::debug(|_| {}),
        ));
        let library = modules::EMBEDDED_MODULES
            .iter()
            .enumerate()
            .map(|(index, (path, text))| {
                let source = index as u32 + 1;
                let (mut bindings, errors) = parse_module(text, source).unwrap();
                assert!(errors.is_empty(), "parse errors in {}", path);
                bindings.insert(0, open("__prelude"));
                LibraryModule {
                    path: path.to_string(),
                    text,
                    source,
                    bindings,
                }
            })
            .collect();
        Scope {
            prelude,
            library,
            program: vec![open("__prelude"), open("std")],
        }
    }

    /// Returns the bindings of the module at `path`.
    pub fn resolve(&self, path: &[String]) -> Result<&[SourceBinding<u32, u32>], Error<u32>> {
        let key = path.join(".");
        if key == "__prelude" {
            return Ok(&self.prelude);
        }
        self.library
            .iter()
            .find(|module| module.path == key)
            .map(|module| &module.bindings[..])
            .ok_or_else(|| Error::eval_here(format!("no module {}", key)))
    }
}

/// Whether a runtime failure is one the sort lattice cannot see — the declared
/// residue, which a clean program is allowed to end in.
pub fn declared_residue<S>(error: &Error<S>) -> bool {
    error.kind() == ErrorKind::Eval
}

/// Returns the runtime sort of an evaluated value, or `None` for non-numerics.
pub fn runtime_sort(expr: &Expr<u32, u32>) -> Option<Sort> {
    match expr {
        Expr::Waveform(Waveform::Const(value)) => Some(if value.fract() == 0.0 {
            Sort::INT
        } else {
            Sort::NON_INT_ONLY
        }),
        Expr::Waveform(_) => Some(Sort::NON_CONST_WAVE),
        Expr::Seq { .. } => Some(Sort::SEQ),
        _ => None,
    }
}

/// A deterministic random source (xorshift).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() % xs.len() as u64) as usize]
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Returns the seed and program count for a sweep, from `SEED` and `COUNT`.
pub fn sweep_bounds(default_count: usize) -> (u64, usize) {
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12345);
    let count = std::env::var("COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_count);
    (seed, count)
}
