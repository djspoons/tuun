// Timing harness for agents/main-thread/tickets/07-evaluation-caching.md
// (re-measured section): eval_timing.rs ported to the processing-thread API
// (processing::process / processing::load); see README.md.
use std::time::Instant;
use std::{env, fs, path::PathBuf};
use tuun::environment::Environment;
use tuun::processing;
use tuun::programs::ProgramSet;
use tuun::prof;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn dump(label: &str) {
    eprintln!("  -- prof [{label}] (key: total ms / max ms / count)");
    for (k, (tot, max, n)) in prof::take() {
        eprintln!("     {k:45} {tot:10.2} {max:10.2} {n:6}");
    }
}

fn nonempty(set: &ProgramSet) -> Vec<usize> {
    (0..set.programs().len())
        .filter(|&i| !set.programs()[i].is_empty())
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let src_path = PathBuf::from(&args[1]);
    let lib = PathBuf::from(&args[2]);
    let new_env = || Environment::new(44100, 120, lib.clone());
    let source = || fs::read_to_string(&src_path).unwrap();

    eprintln!("== 1. read + from_source");
    for run in 0..3 {
        let t = Instant::now();
        let (set, _) = ProgramSet::from_source(source(), src_path.clone()).unwrap();
        eprintln!("  run {run}: {:.2} ms, programs {}", ms(t), nonempty(&set).len());
    }
    prof::take();

    // 2. per program check vs evaluate, warm env.
    let (set, _) = ProgramSet::from_source(source(), PathBuf::new()).unwrap();
    {
        let env = new_env();
        let _ = processing::load(&env, source(), PathBuf::new(), false);
        prof::take();
        eprintln!("== 2. per program check / evaluate / context (warm env)");
        let (mut ct, mut et, mut xt) = (0.0, 0.0, 0.0);
        let mut nfail = 0;
        for i in nonempty(&set) {
            let input = set.evaluation_input(i);
            let t = Instant::now();
            let findings = env.check_program(&input);
            let c = ms(t);
            let t = Instant::now();
            let r = env.evaluate_program(&input);
            let e = ms(t);
            let t = Instant::now();
            let _ = env.program_context(&set, i);
            let x = ms(t);
            if findings.iter().any(|d| d.severity == tuun::diagnostics::Severity::Error) || r.is_err() {
                nfail += 1;
                eprintln!("  program {} FAIL", set.display_name(i));
            }
            eprintln!("  {:24} check {c:9.2}  eval {e:9.2}  context {x:9.2}", set.display_name(i));
            ct += c;
            et += e;
            xt += x;
        }
        eprintln!("  TOTAL check {ct:.1} ms, eval {et:.1} ms, context {xt:.1} ms, failures {nfail}");
        prof::take();
    }

    // 3. reload equivalent: processing::load, cold and warm, precompute off and on.
    for precompute in [false, true] {
        for pass in ["cold env", "warm env"] {
            let env = new_env();
            if pass == "warm env" {
                let _ = processing::load(&env, source(), src_path.clone(), precompute);
            }
            prof::take();
            let t = Instant::now();
            let loaded = processing::load(&env, source(), src_path.clone(), precompute).unwrap();
            eprintln!(
                "== 3. load ({pass}, precompute={precompute}): {:.1} ms (failed {})",
                ms(t),
                loaded.failed.len()
            );
            dump(pass);
        }
    }

    // 4. leaving Edit mode: process one program, warm env.
    let progs = nonempty(&set);
    let env = new_env();
    let _ = processing::load(&env, source(), PathBuf::new(), false);
    for &target in [progs[0], progs[progs.len() / 2], *progs.last().unwrap()].iter() {
        prof::take();
        for run in 0..3 {
            let t = Instant::now();
            let r = processing::process(&env, &set.evaluation_input(target), true);
            eprintln!(
                "== 4. process {} run {run}: {:.1} ms ok={}",
                set.display_name(target),
                ms(t),
                r.is_ok()
            );
        }
        dump("edit");
    }
}
