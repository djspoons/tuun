//! Generated-program sweeps of the checker against evaluation.
//!
//! A deterministic generator plus three sweeps over what it produces: a
//! canary pinning which programs check clean, a soundness search, and a
//! false-positive search. The generator is a regression net rather than a
//! discovery tool: it finds instances of shapes it already knows how to build,
//! so a new finding usually means adding a case to `gen_expr` first.

use tuun::builtins;
use tuun::eval;
use tuun::expr::{Binding, Error, ErrorKind, Expr, Pattern, SourceBinding, SourceExpr};
use tuun::infer::{ModuleCache, check_program};
use tuun::modules;
use tuun::parser::{parse_module, parse_program};
use tuun::signatures::signature;
use tuun::types::{Refinement, Sort, Type};
use tuun::waveform::Waveform;

/// The bindings generated programs check and evaluate under: the built-ins
/// and std, as a program that opens std sees them.
struct Scope {
    /// The built-ins plus `tempo` and `sample_rate`.
    prelude: Vec<SourceBinding<u32, u32>>,
    /// The embedded std module, opening the prelude.
    std: Vec<SourceBinding<u32, u32>>,
    /// The bindings a program is checked under: it opens the prelude, then
    /// std.
    program: Vec<SourceBinding<u32, u32>>,
}

impl Scope {
    fn new() -> Scope {
        let open = |path: &str| Binding::Open(vec![path.to_string()]).into();
        let mut prelude = Vec::new();
        builtins::add_bindings(&mut prelude);
        for (name, value) in [("tempo", 120.0), ("sample_rate", 44100.0)] {
            prelude.push(SourceBinding::definition(
                Pattern::Identifier(name.to_string()),
                SourceExpr::float(value),
            ));
        }
        let (_, text) = modules::EMBEDDED_MODULES
            .iter()
            .find(|(path, _)| *path == "std")
            .expect("std is embedded");
        let (mut std, errors) = parse_module(text, 1u32).expect("std parses");
        assert!(errors.is_empty(), "std has parse errors");
        std.insert(0, open("__prelude"));
        Scope {
            prelude,
            std,
            // Opening std adds about 14 ms to every `eval::evaluate` (measured
            // on `1 + 2`), since evaluation has no counterpart to the checker's
            // `ModuleCache`; that is about half of `soundness_fuzz`'s time.
            program: vec![open("__prelude"), open("std")],
        }
    }

    /// Returns the bindings of the module at `path`.
    fn resolve(&self, path: &[String]) -> Result<&[SourceBinding<u32, u32>], Error<u32>> {
        match path.join(".").as_str() {
            "__prelude" => Ok(&self.prelude),
            "std" => Ok(&self.std),
            key => Err(Error::eval_here(format!("no module {}", key))),
        }
    }
}

/// Whether a runtime failure is one the sort lattice cannot see — the declared
/// residue, which a clean program is allowed to end in.
fn declared_residue<S>(error: &Error<S>) -> bool {
    error.kind() == ErrorKind::Eval
}

/// Returns the runtime sort of an evaluated value, or `None` for non-numerics.
fn runtime_sort(expr: &Expr<u32, u32>) -> Option<Sort> {
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
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() % xs.len() as u64) as usize]
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Returns the seed and program count for a sweep, from `SEED` and `COUNT`.
fn sweep_bounds(default_count: usize) -> (u64, usize) {
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
    (12345, 350, 12924745280370670482),
    (777, 350, 12246305972029091934),
    (20260901, 350, 9767260543059777376),
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
/// SEED=7 COUNT=50000 cargo test --test checker_fuzz soundness_fuzz -- --ignored --nocapture
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
