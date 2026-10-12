//! The checker run over the embedded library, against evaluation.
//!
//! Every module checks under the real prelude and its real dependencies, so a
//! change to a signature, a built-in, or a library module can move these as
//! well as a checker change.

mod common;

use common::{LibraryModule, Scope, declared_residue, runtime_sort};
use tuun::eval;
use tuun::expr::{Binding, Pattern, SourceBinding};
use tuun::infer::{ModuleCache, check_program};
use tuun::parser::parse_program;
use tuun::signatures::signature;
use tuun::types::{Refinement, Sort, Type};

/// The span source id of the root program each module is checked under,
/// apart from every module's own.
const ROOT_SOURCE: u32 = 9999;

/// Returns the errors in `module`'s own text from checking it as a program's
/// bindings, each rendered against that text.
///
/// Errors from dependencies re-typed along the way are left to their own
/// module's check.
fn check_module(
    scope: &Scope,
    module: &LibraryModule,
    cache: &mut ModuleCache<u32>,
) -> Vec<String> {
    let expr = parse_program::<u32, _>("0", ROOT_SOURCE).unwrap();
    let resolve = |path: &[String]| scope.resolve(path);
    check_program(resolve, &module.bindings, signature, &expr, None, cache)
        .0
        .iter()
        .filter(|error| error.source() == Some(module.source))
        .map(|error| error.display_with_source(module.text))
        .collect()
}

/// Returns a closed expression inhabiting `sort`, preferring the most general
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

/// Returns the type of each definition `module` exports, as a program that
/// `use`s the module sees it.
fn exports(scope: &Scope, module: &LibraryModule) -> Vec<(String, Type)> {
    let path: Vec<String> = module.path.split('.').map(str::to_string).collect();
    let name = path.last().unwrap().clone();
    let bindings: Vec<SourceBinding<u32, u32>> = vec![Binding::Use(path).into()];
    let expr = parse_program::<u32, _>(&name, ROOT_SOURCE).unwrap();
    let resolve = |path: &[String]| scope.resolve(path);
    let (_, ty) = check_program(
        resolve,
        &bindings,
        signature,
        &expr,
        None,
        &mut ModuleCache::default(),
    );
    let Type::Module(members) = ty else {
        panic!("`use {}` is not a module: {}", module.path, ty);
    };
    members
}

/// The soundness theorem as a regression test, over the library corpus: no
/// clean program may evaluate to an error outside the declared residue.
///
/// Covers every module as a program root, and every exported function
/// applied once per table conjunct at representative arguments — the
/// tabulated analog of the builtin conformance test, checking result sorts
/// against the conjuncts.
#[test]
fn differential_library_agreement() {
    let scope = Scope::new();
    let resolve = |path: &[String]| scope.resolve(path);
    let mut calls = 0;
    let mut true_positives = 0;
    let mut residue = 0;
    // Every module as a program root: its bindings must evaluate, and
    // checking must agree.
    for module in &scope.library {
        let expr = parse_program::<u32, _>("0", ROOT_SOURCE).unwrap();
        let clean = check_program(
            resolve,
            &module.bindings,
            signature,
            &expr,
            None,
            &mut ModuleCache::default(),
        )
        .0
        .is_empty();
        let evaluated = eval::evaluate(resolve, &module.bindings, expr);
        if let (true, Err(error)) = (clean, &evaluated) {
            assert!(
                declared_residue(error),
                "soundness: module {} is clean but evaluates to: {}",
                module.path,
                error.message()
            );
        }
        calls += 1;
    }
    // Every exported function applied once per all-numeric table conjunct.
    // The calls batch into one synthesized module per library module —
    // one check and one evaluation for all of them — with a
    // distinct source id per call for attribution.
    for module in &scope.library {
        let bindings = &module.bindings;
        // (name to bind, call text, source id, declared result sort)
        let mut applied: Vec<(String, String, u32, Option<Sort>)> = Vec::new();
        for (name, ty) in exports(&scope, module) {
            let body = match &ty {
                Type::Forall(_, body) => &**body,
                other => other,
            };
            let conjuncts: Vec<&Type> = match body {
                Type::And(conjuncts) => conjuncts.iter().collect(),
                other => vec![other],
            };
            for conjunct in conjuncts.iter() {
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
        let expr = parse_program::<u32, _>("0", ROOT_SOURCE).unwrap();
        // A closure of its own: the checker ties the bindings' lifetime to
        // the modules `resolve` returns, and `extended` is local.
        let errors = check_program(
            |path: &[String]| scope.resolve(path),
            &extended,
            signature,
            &expr,
            None,
            &mut ModuleCache::default(),
        )
        .0;
        let flagged: Vec<u32> = errors.iter().filter_map(|error| error.source()).collect();
        // Classifies one call by evaluating it alone against the
        // module. An eval error's span points into the failing body,
        // not the call, so per-call evaluation is the only reliable
        // attribution; it runs for flagged calls and as the fallback
        // when the batch fails.
        let mut classify = |text: &str, id: u32, declared: Option<Sort>| {
            let expr = parse_program::<u32, _>(text, id).unwrap();
            let evaluated = eval::evaluate(resolve, bindings, expr);
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
                    if let (Some(declared), Some(actual)) = (declared, runtime_sort(&value.expr)) {
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
        match eval::evaluate_bindings(|path: &[String]| scope.resolve(path), &batch) {
            Ok(evaluated) => {
                for (bound, text, id, declared) in &applied {
                    if flagged.contains(id) {
                        continue;
                    }
                    let Some((_, value)) = evaluated.iter().rev().find(|(name, _)| name == bound)
                    else {
                        continue;
                    };
                    if let (Some(declared), Some(actual)) = (declared, runtime_sort(&value.expr)) {
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

/// Pins the checker's verdict on every embedded library module, so
/// signature or inference changes that affect the library surface here.
///
/// A module's report keeps only errors from its own text — dependencies
/// re-typed along the way report under their own entry — and positions
/// come from `display_with_source`, so every pinned error can be read
/// against the .tuun source.
#[test]
fn embedded_modules_pinned_errors() {
    let scope = Scope::new();
    let mut report: Vec<String> = Vec::new();
    for module in &scope.library {
        for error in check_module(&scope, module, &mut ModuleCache::default()) {
            report.push(format!("{}: {}", module.path, error));
        }
    }
    // The refinement lattice checks the entire embedded library without a
    // single error. Anything appearing here needs source-level
    // verification before being accepted as a new pin.
    assert_eq!(report, Vec::<String>::new());
}

/// Prints each embedded module's own errors (run with `--ignored
/// --nocapture`).
#[test]
#[ignore]
fn library_report() {
    let scope = Scope::new();
    for module in &scope.library {
        let own = check_module(&scope, module, &mut ModuleCache::default());
        println!("=== {} ({} errors)", module.path, own.len());
        for error in own {
            println!("  {}", error);
        }
    }
}

/// Returns the number of rounds a timing test runs, from `ROUNDS`.
fn rounds() -> usize {
    std::env::var("ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
}

/// Times repeated checks of the embedded library (run with `--ignored
/// --nocapture`), for comparing a change against the build before it.
///
/// ```text
/// cargo test --release --test checker_library library_timing -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn library_timing() {
    let rounds = rounds();
    let scope = Scope::new();
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..rounds {
        let start = std::time::Instant::now();
        for module in &scope.library {
            std::hint::black_box(check_module(&scope, module, &mut ModuleCache::default()));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        best = best.min(elapsed);
        total += elapsed;
    }
    println!(
        "TIMING best {:.2}ms  mean {:.2}ms  over {} rounds",
        best,
        total / rounds as f64,
        rounds
    );
}

/// Times repeated checks of the embedded library with a shared
/// [`ModuleCache`] (run with `--ignored --nocapture`) — the session-warm
/// counterpart of [`library_timing`].
#[test]
#[ignore]
fn library_timing_warm() {
    let rounds = rounds();
    let scope = Scope::new();
    let mut cache = ModuleCache::default();
    let mut best = f64::MAX;
    let mut total = 0.0;
    for round in 0..rounds {
        let start = std::time::Instant::now();
        for module in &scope.library {
            std::hint::black_box(check_module(&scope, module, &mut cache));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if round == 0 {
            println!("TIMING cold first round {:.2}ms", elapsed);
            continue;
        }
        best = best.min(elapsed);
        total += elapsed;
    }
    println!(
        "TIMING warm best {:.2}ms  mean {:.2}ms  over {} rounds",
        best,
        total / (rounds - 1) as f64,
        rounds - 1
    );
}
