// Timing harness for agents/main-thread/tickets/06-evaluation.md; see README.md.
use std::time::Instant;
use std::{env, fs, path::PathBuf};
use tuun::environment::Environment;
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

fn reload_loop(env: &Environment, set: &mut ProgramSet) -> (f64, usize) {
    let t = Instant::now();
    let mut failed = 0;
    for i in nonempty(set) {
        if set.evaluate_and_record(env, i).is_err() {
            failed += 1;
        }
    }
    (ms(t), failed)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let src_path = PathBuf::from(&args[1]);
    let lib = PathBuf::from(&args[2]);
    let new_env = || Environment::new(44100, 120, lib.clone());

    if let Ok(k) = env::var("ONE") {
        let env = new_env();
        let (mut s, _) =
            ProgramSet::from_source(fs::read_to_string(&src_path).unwrap(), PathBuf::new()).unwrap();
        let i = nonempty(&s)[k.parse::<usize>().unwrap()];
        for _ in 0..2 {
            let t = Instant::now();
            let _ = s.evaluate_and_record(&env, i);
            eprintln!("ONE {} {:.1} ms", s.display_name(i), ms(t));
        }
        return;
    }
    // 1. read + parse
    eprintln!("== 1. read + from_source");
    for run in 0..3 {
        let t = Instant::now();
        let source = fs::read_to_string(&src_path).unwrap();
        let read = ms(t);
        let t = Instant::now();
        let (set, warn) = ProgramSet::from_source(source, src_path.clone()).unwrap();
        let parse = ms(t);
        eprintln!(
            "  run {run}: read {read:.3} ms, from_source {parse:.2} ms, programs {}, bindings-warning '{warn}'",
            nonempty(&set).len()
        );
    }
    dump("1");

    // 2. per program check vs evaluate (cold env, then warm env)
    let (mut set, _) =
        ProgramSet::from_source(fs::read_to_string(&src_path).unwrap(), PathBuf::new()).unwrap();
    let skip2 = env::var("SKIP2").is_ok();
    for pass in ["cold env", "warm env"] {
        if skip2 { break; }
        let env = if pass == "cold env" { new_env() } else { new_env() };
        if pass == "warm env" {
            // warm caches with a full pass first
            reload_loop(&env, &mut set);
            prof::take();
        }
        eprintln!("== 2. per program check / evaluate ({pass})");
        let (mut ct, mut et) = (0.0, 0.0);
        let (mut cmax, mut emax) = ((0.0, 0), (0.0, 0));
        let mut nfail = 0;
        for i in nonempty(&set) {
            let t = Instant::now();
            let findings = env.check_program(&set, i);
            let c = ms(t);
            let t = Instant::now();
            let r = env.evaluate_program(&set, i);
            let e = ms(t);
            if !findings.is_empty() || r.is_err() {
                nfail += 1;
                eprintln!(
                    "  program {} FAIL findings={:?} eval_err={:?}",
                    set.display_name(i),
                    findings.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
                    r.as_ref().err().map(|v| v.iter().map(|d| d.to_string()).collect::<Vec<_>>())
                );
            }
            eprintln!("  {:24} check {c:9.2}  eval {e:9.2}", set.display_name(i));
            ct += c;
            et += e;
            if c > cmax.0 {
                cmax = (c, i);
            }
            if e > emax.0 {
                emax = (e, i);
            }
        }
        eprintln!(
            "  TOTAL check {ct:.1} ms (max {:.1} @ {}), eval {et:.1} ms (max {:.1} @ {}), failures {nfail}",
            cmax.0,
            set.display_name(cmax.1),
            emax.0,
            set.display_name(emax.1)
        );
        dump(pass);
    }

    // Full ReloadSource-equivalent: reload_from_disk + evaluate_and_record loop
    for pass in ["cold env", "warm env"] {
        if skip2 && pass == "cold env" { continue; }
        let env = new_env();
        let (base, _) =
            ProgramSet::from_source(fs::read_to_string(&src_path).unwrap(), src_path.clone())
                .unwrap();
        if pass == "warm env" {
            let mut s2 = base.reload_from_disk().unwrap().0;
            reload_loop(&env, &mut s2);
        }
        prof::take();
        let t = Instant::now();
        let (mut s, _) = base.reload_from_disk().unwrap();
        let load = ms(t);
        let (eval_ms, failed) = reload_loop(&env, &mut s);
        eprintln!(
            "== ReloadSource equivalent ({pass}): reload_from_disk {load:.2} ms + evaluate_and_record loop {eval_ms:.1} ms (failed {failed})"
        );
        dump(pass);
    }

    // 3. splice
    let tmp = std::env::temp_dir().join(format!("tuun_eval_timing_{}.tuun", std::process::id()));
    fs::copy(&src_path, &tmp).unwrap();
    let progs = nonempty(&set);
    for &target in [progs[progs.len() / 2], *progs.last().unwrap()].iter() {
        for with_path in [false, true] {
            let p = if with_path { tmp.clone() } else { PathBuf::new() };
            let (mut s, _) = ProgramSet::from_source(fs::read_to_string(&tmp).unwrap(), p).unwrap();
            prof::take();
            let mut times = vec![];
            for _ in 0..5 {
                let text = format!("{} ", s.programs()[target].text().trim_end());
                s.program_mut(target).unwrap().set_text(text);
                let t = Instant::now();
                s.splice(target).unwrap();
                times.push(ms(t));
            }
            eprintln!(
                "== 3. splice program {} write={with_path}: runs {:?}",
                s.display_name(target),
                times.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>()
            );
            dump("splice");
        }
    }
    let _ = fs::remove_file(&tmp);

    // 4. leaving edit mode: evaluate_and_record of one program, first vs second run
    for &target in [progs[0], progs[progs.len() / 2], *progs.last().unwrap()].iter() {
        let env = new_env();
        let (mut s, _) =
            ProgramSet::from_source(fs::read_to_string(&src_path).unwrap(), PathBuf::new()).unwrap();
        prof::take();
        for run in 0..3 {
            let t = Instant::now();
            let r = s.evaluate_and_record(&env, target);
            eprintln!(
                "== 4. evaluate_and_record {} run {run}: {:.1} ms ok={}",
                s.display_name(target),
                ms(t),
                r.is_ok()
            );
            dump("edit");
        }
    }
}
