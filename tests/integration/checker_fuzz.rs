//! Generated-program sweeps of the checker against evaluation.
//!
//! A deterministic generator plus three sweeps over what it produces: a canary
//! pinning which programs check clean, a soundness search, and a false-positive
//! search. The generator is a regression net rather than a discovery tool: it
//! finds instances of shapes it already knows how to build, so a new finding
//! usually means adding a case to `gen_expr` first.
//!
//! An additional sweep, `wide_function_fuzz`, generates its own programs:
//! functions too wide to tabulate.

use crate::common::{Rng, Scope, declared_residue, runtime_sort, sweep_bounds};
use tuun::eval;
use tuun::infer::{ModuleCache, check_program};
use tuun::parser::parse_program;
use tuun::signatures::signature;
use tuun::types::{Refinement, Type};

const ATOMS: &[&str] = &[
    "1",
    "2",
    "0.5",
    "time",
    "(time | seq(time - 1))",
    "true",
    "\"s\"",
    "[1, 2]",
    "[time]",
    "(1, time)",
    "x",
    "y",
    "sqrt",
    "cos",
    "log",
    "g",
    "h",
    "k",
];
const BIN: &[&str] = &["+", "-", "*", "/", "&", "\\", "<", ">", "==", "!="];
const UN1: &[&str] = &[
    "sqrt",
    "exp",
    "cos",
    "unseq()",
    "-",
    "fin(4)",
    "capture(\"c\")",
];
const FN1: &[&str] = &["sqrt", "exp", "cos", "g", "h", "k"];
const BUILTIN2: &[&str] = &["log", "phase", "reset", "append", "pow"];

fn gen_expr(rng: &mut Rng, depth: usize) -> String {
    if depth == 0 {
        return rng.pick(ATOMS).to_string();
    }
    let d = depth - 1;
    match rng.below(34) {
        0 | 1 => format!(
            "({} {} {})",
            gen_expr(rng, d),
            rng.pick(BIN),
            gen_expr(rng, d)
        ),
        2 => {
            let f = rng.pick(UN1);
            if f == "-" {
                format!("(- {})", gen_expr(rng, d))
            } else {
                format!("({} | {})", gen_expr(rng, d), f)
            }
        }
        3 => format!(
            "{}({}, {})",
            rng.pick(BUILTIN2),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        4 => format!(
            "(if {} then {} else {})",
            rng.pick(&["true", "false"]),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        5 => format!("(fn(x) => {})({})", gen_expr(rng, d), gen_expr(rng, d)),
        6 => format!(
            "(fn(x) => fn(y) => {})({})({})",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        7 => format!("nth(1, [{}, {}])", gen_expr(rng, d), gen_expr(rng, d)),
        8 => format!("map(fn(x) => {}, [{}])", gen_expr(rng, d), gen_expr(rng, d)),
        9 => format!(
            "reduce(fn(y, x) => {}, {}, [{}])",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        10 => format!(
            "(let f = fn(x) => {} in f({}))",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        // named parameters: omitted, supplied, and mixed with positionals
        11 | 12 => format!(
            "(let f = fn(k = {}) => {} in f())",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        13 | 14 => format!(
            "(let f = fn(x, k = {}) => {} in f({}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        15 | 16 => format!(
            "(let f = fn(x, k = {}) => {} in f({}, k = {}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        17 => format!(
            "(let f = fn(a = {}, b = {}) => {} in f())",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        18 => format!(
            "(let f = fn(a = {}, b = {}) => {} in f(b = {}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        19 => format!(
            "(let f = fn(a) => fn(b) => {} in f({})({}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        20 => format!(
            "(let ap = fn(f, v) => f(v) in ap(fn(x) => {}, {}))",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        21 => format!(
            "(let c = fn(f, g2) => fn(v) => f(g2(v)) in c(fn(x) => {}, fn(x) => {})({}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        22 => format!(
            "(fn((a, b)) => {})(({}, {}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        23 => format!("<[{}, {}]>", gen_expr(rng, d), gen_expr(rng, d)),
        24 => format!("{{[{}, {}]}}", gen_expr(rng, d), gen_expr(rng, d)),
        25 => format!(
            "(if {} then {} else {})({})",
            rng.pick(&["true", "false"]),
            rng.pick(FN1),
            rng.pick(FN1),
            gen_expr(rng, d)
        ),
        26 => format!(
            "nth(1, [{}, {}])({})",
            rng.pick(FN1),
            rng.pick(FN1),
            gen_expr(rng, d)
        ),
        // named parameters whose *body* uses the parameter, so a
        // supplied argument can disagree with the default's sort
        28 => format!(
            "(let f = fn(k = {}) => k in f(k = {}))",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        29 => format!(
            "(let f = fn(k = {}) => nth(0, k) in f(k = {}))",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        30 => format!(
            "(let f = fn(x, k = {}) => (x, k) in f({}, k = {}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        // partial application bound to a name: the inner lambda is one
        // ABS never reached before S2
        31 => format!(
            "(let g = (fn(a) => fn(b) => {})({}) in g({}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        32 => format!(
            "(let f = fn(a) => fn(b) => {} in let g = f({}) in g({}))",
            gen_expr(rng, d),
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        33 => format!(
            "(let f = fn(a, b, c, d2, e2) => {} in f({}, 1, 1, 1, 1))",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
        _ => format!(
            "unfold(fn(v) => {}, {}, 2)",
            gen_expr(rng, d),
            gen_expr(rng, d)
        ),
    }
}

/// One generated program, wrapped in the prelude the generator's atoms
/// assume.
///
/// The prelude's functions are written as a user would: `g` and `h` take their
/// types from built-ins with several conjuncts, and `k` from one with a single
/// arrow.
fn generated(rng: &mut Rng) -> String {
    // A consumer around the expression gives its *result* a use, so a
    // wrong result type is caught rather than merely inferred.
    const CONSUMERS: &[&str] = &[
        "{}",
        "nth({}, [1, 2])",
        "sqrt({})",
        "unseq()({})",
        "<[{}]>",
        "({} \\ 1)",
        "log({}, 2)",
        "fixed([{}])",
        "unfold(fn(v) => v, 1, {})",
        "({} + (time | seq(time - 1)))",
        "fin({})(time)",
        "{{[{}]}}",
    ];
    let depth = 1 + rng.below(3);
    let inner = gen_expr(rng, depth);
    let inner = CONSUMERS[rng.below(CONSUMERS.len())].replacen("{}", &inner, 1);
    format!(
        "let x = 1 in let y = time in let g = fn(v) => v + 1 in \
         let h = fn(v) => v | fin(4) in let k = fn(v) => append(v, 0) in {}",
        inner
    )
}

/// The seeds and program count the accept-vector canary is pinned at.
/// Three seeds for shape diversity, sized to stay near a second. A hash
/// trips on a single differing program, so this catches a boundary move of
/// roughly one program in a thousand; `soundness_fuzz` and
/// `false_positive_fuzz` are where a finer sweep belongs.
const CANARY: [(u64, usize, u64); 3] = [
    (12345, 350, 1170461391898692706),
    (777, 350, 12246305972029091934),
    (20260901, 350, 9300518567311359572),
];

/// Returns an FNV-1a hash of which generated programs check clean.
///
/// FNV rather than [`std::collections::hash_map::DefaultHasher`], whose
/// output is not guaranteed stable across toolchain versions — a pinned
/// constant needs a hash that is.
fn accept_hash(seed: u64, count: usize) -> u64 {
    let mut rng = Rng(seed);
    let scope = Scope::new();
    let resolve = |path: &[String]| scope.resolve(path);
    let mut cache = ModuleCache::default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for _ in 0..count {
        let text = generated(&mut rng);
        let accepted = match parse_program::<u32, _>(&text, 0u32) {
            Ok(expr) => check_program(resolve, &scope.program, signature, &expr, None, &mut cache)
                .0
                .is_empty(),
            Err(_) => continue,
        };
        hash ^= u64::from(accepted);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Pins *which* generated programs typecheck, so a change that alters the
/// set has to say so.
///
/// This is a canary, not a correctness claim: the hash changing does not
/// mean a change is wrong, only that it moved the boundary. Programs check
/// against the real built-ins and std, so a change to a signature or a std
/// definition the generator reaches can move it as well as a checker change.
/// When it fires, compare the two builds program by program — a fix should
/// move the boundary in one direction only — and then update the constant
/// here along with the entry that explains the move.
#[test]
fn the_accepted_set_is_unchanged() {
    let actual: Vec<(u64, usize, u64)> = CANARY
        .iter()
        .map(|(seed, count, _)| (*seed, *count, accept_hash(*seed, *count)))
        .collect();
    assert_eq!(
        actual,
        CANARY.to_vec(),
        "the set of programs that typecheck has changed; see this test's doc comment"
    );
}

/// Searches for programs the checker accepts and evaluation then fails.
///
/// This is the soundness claim stated as a test: a program that checks
/// clean must evaluate, or fail for one of the reasons `declared_residue`
/// names. A declared sort that the value does not inhabit counts too.
///
/// Reports one program per distinct failure so a single cause does not
/// fill the output. Run with `--ignored --nocapture`, and set `SEED` and
/// `COUNT` to sweep further:
///
/// ```text
/// SEED=7 COUNT=50000 cargo test --test integration checker_fuzz::soundness_fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn soundness_fuzz() {
    let (seed, count) = sweep_bounds(20000);
    let mut rng = Rng(seed);
    let scope = Scope::new();
    let resolve = |path: &[String]| scope.resolve(path);
    let mut cache = ModuleCache::default();
    let (mut found, mut clean) = (0, 0);
    let mut kinds: Vec<String> = Vec::new();
    for _ in 0..count {
        let text = generated(&mut rng);
        let Ok(expr) = parse_program::<u32, _>(&text, 0u32) else {
            continue;
        };
        let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_program(resolve, &scope.program, signature, &expr, None, &mut cache)
        }));
        let Ok((errors, ty)) = checked else {
            // A panic may leave the cache half-built.
            cache = ModuleCache::default();
            found += 1;
            println!("CHECKER-PANIC[{}] {}", found, text);
            if found > 30 {
                break;
            }
            continue;
        };
        if !errors.is_empty() {
            continue;
        }
        clean += 1;
        // The declared sort, for comparison against the value's own.
        let declared = match ty {
            Type::Numeric(Refinement::Ground(sort)) => Some(sort),
            _ => None,
        };
        let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eval::evaluate(resolve, &scope.program, expr)
        }));
        let (kind, detail) = match evaluated {
            Err(_) => ("EVAL-PANIC".to_string(), String::new()),
            Ok(Err(error)) => {
                if declared_residue(&error) {
                    continue;
                }
                ("eval-error".to_string(), error.message().to_string())
            }
            Ok(Ok(value)) => {
                let Some(declared) = declared else { continue };
                let Some(actual) = runtime_sort(&value.expr) else {
                    continue;
                };
                if actual.is_subset(declared) {
                    continue;
                }
                (
                    "sort-escape".to_string(),
                    format!("declared {} but evaluated to {}", declared, actual),
                )
            }
        };
        let signature = format!("{}: {}", kind, detail);
        if kinds.contains(&signature) {
            continue;
        }
        kinds.push(signature);
        found += 1;
        println!("UNSOUND[{}] ({}) {}\n    {}", found, kind, text, detail);
        if found > 30 {
            break;
        }
    }
    println!(
        "soundness: {} of {} accepted, {} distinct classes",
        clean, count, found
    );
    assert_eq!(found, 0, "the checker accepted a program that does not run");
}

/// Searches for programs the checker rejects and evaluation accepts.
///
/// A hit is not necessarily a defect — the checker is deliberately
/// stricter than one run of the program, since it answers for every run —
/// but each one is a claim worth reading. Reports one program per distinct
/// set of messages, and does not assert: this is an instrument, not a
/// contract.
#[test]
#[ignore]
fn false_positive_fuzz() {
    let (seed, count) = sweep_bounds(20000);
    let mut rng = Rng(seed);
    let scope = Scope::new();
    let resolve = |path: &[String]| scope.resolve(path);
    let mut cache = ModuleCache::default();
    let (mut found, mut rejected) = (0, 0);
    let mut kinds: Vec<String> = Vec::new();
    for _ in 0..count {
        let text = generated(&mut rng);
        let Ok(expr) = parse_program::<u32, _>(&text, 0u32) else {
            continue;
        };
        let Ok((errors, _)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_program(resolve, &scope.program, signature, &expr, None, &mut cache)
        })) else {
            // A panic may leave the cache half-built.
            cache = ModuleCache::default();
            continue;
        };
        if errors.is_empty() {
            continue;
        }
        rejected += 1;
        let messages: Vec<String> = errors.iter().map(|e| e.message().to_string()).collect();
        let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eval::evaluate(resolve, &scope.program, expr)
        }));
        if !matches!(evaluated, Ok(Ok(_))) {
            continue;
        }
        let signature = format!("{:?}", messages);
        if kinds.contains(&signature) {
            continue;
        }
        kinds.push(signature);
        found += 1;
        println!("FALSE-POS[{}] {}\n    {:?}", found, text, messages);
        if found > 30 {
            break;
        }
    }
    println!(
        "false positives: {} of {} rejected, {} distinct classes",
        rejected, count, found
    );
}

/// Searches for unsoundness among functions too wide to tabulate.
///
/// The invariant is "refused *or* sound", which holds under any budget: a
/// definition past the checker's coarse-tabulation budget is refused, and
/// whatever falls inside it is tabulated and still has to run. Run it whenever
/// either budget or the refusal changes.
///
/// This is the harness that showed refusing was necessary. With the refusal
/// disabled, the summary the checker fell back on declared sorts the values
/// did not inhabit: three escapes in 450,000 programs across nine seeds, all
/// of the form "declared `waveform or seq`, evaluated to a constant".
///
/// ```text
/// SEED=7 COUNT=50000 cargo test --release --test integration checker_fuzz::wide_function_fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn wide_function_fuzz() {
    const OPS: &[&str] = &["+", "-", "*", "/", "&"];
    const ARGS: &[&str] = &["1", "2", "0.5", "time", "(time | seq(time - 1))"];
    let (seed, count) = sweep_bounds(20000);
    let mut rng = Rng(seed);
    let scope = Scope::new();
    let resolve = |path: &[String]| scope.resolve(path);
    let mut cache = ModuleCache::default();
    let (mut clean, mut found, mut widest) = (0usize, 0usize, 0usize);
    let mut seqs = 0usize;
    let mut kinds: Vec<String> = Vec::new();
    for _ in 0..count {
        // 10..14 parameters: past the coarse split's 2^9 budget.
        let arity = 10 + rng.below(5);
        let names: Vec<String> = (0..arity).map(|i| format!("p{}", i)).collect();
        // A body that uses every parameter numerically, so all of them
        // tabulate (and so the width is real). Each parameter enters
        // through its own randomly chosen operator and filter, so the
        // relation the body imposes is not uniform across positions.
        const FILTERS: &[&str] = &["fin(4)", "unseq()", "capture(\"c\")", "cos", "sqrt"];
        let wrap = |rng: &mut Rng, term: String| match rng.below(14) {
            0 => format!("({} | {})", term, rng.pick(FILTERS)),
            1 => format!("(- {})", term),
            2 => format!("({} {} {})", term, rng.pick(OPS), rng.pick(ARGS)),
            _ => term,
        };
        let mut body = wrap(&mut rng, names[0].clone());
        for name in &names[1..] {
            let term = wrap(&mut rng, name.clone());
            body = format!("({} {} {})", body, rng.pick(OPS), term);
        }
        // A consumer gives the *result* a use, so a wrong result sort is
        // caught rather than merely inferred.
        const CONSUMERS: &[&str] = &[
            "{}",
            "({} \\ 1)",
            "nth({}, [1, 2])",
            "<[{}]>",
            "sqrt({})",
            "unseq()({})",
            "({} + (time | seq(time - 1)))",
            "fin({})(time)",
            "{{[{}]}}",
        ];
        // Half the time the body stands alone: a consumer that rejects
        // everything would leave nothing accepted to test.
        if rng.below(2) == 0 {
            body = CONSUMERS[rng.below(CONSUMERS.len())].replacen("{}", &body, 1);
        }
        let args: Vec<&str> = (0..arity).map(|_| rng.pick(ARGS)).collect();
        let text = format!(
            "let f = fn({}) => {} in f({})",
            names.join(", "),
            body,
            args.join(", ")
        );
        widest = widest.max(arity);
        let Ok(expr) = parse_program::<u32, _>(&text, 0u32) else {
            continue;
        };
        let (errors, ty) =
            check_program(resolve, &scope.program, signature, &expr, None, &mut cache);
        if !errors.is_empty() {
            continue;
        }
        clean += 1;
        if text.contains("seq") {
            seqs += 1;
        }
        let declared = match ty {
            Type::Numeric(Refinement::Ground(sort)) => Some(sort),
            _ => None,
        };
        let (kind, detail) = match eval::evaluate(resolve, &scope.program, expr) {
            Err(error) => {
                if declared_residue(&error) {
                    continue;
                }
                ("eval-error".to_string(), error.message().to_string())
            }
            Ok(value) => {
                let Some(declared) = declared else { continue };
                let Some(actual) = runtime_sort(&value.expr) else {
                    continue;
                };
                if actual.is_subset(declared) {
                    continue;
                }
                (
                    "sort-escape".to_string(),
                    format!("declared {} but evaluated to {}", declared, actual),
                )
            }
        };
        let signature = format!("{}: {}", kind, detail);
        if kinds.contains(&signature) {
            continue;
        }
        kinds.push(signature);
        found += 1;
        println!(
            "WIDE-UNSOUND[{}] ({}) {}\n    {}",
            found, kind, text, detail
        );
        if found > 30 {
            break;
        }
    }
    println!(
        "wide: {} of {} accepted ({} of them pass a seq, widest arity {}), {} distinct classes",
        clean, count, seqs, widest, found
    );
    assert_eq!(
        found, 0,
        "a function too wide to tabulate was accepted and does not run"
    );
}
