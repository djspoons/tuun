# Caching evaluated modules and bindings

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Should processing cache what it now evaluates again for every program, so results arrive sooner? Three costs dominate:
- the exports of each opened or used module (and its dependencies);
- the named bindings earlier in the file, which makes a reload quadratic in the number of named programs;
- the second evaluation of `on_beats` programs in `evaluated_waveform`.

Measured in [Where program processing runs](06-evaluation.md): evaluation is about 93–95% of a reload, and nearly all of the time to leave Edit mode. This shortens the "Processing…" wait, not input latency.

## Out of scope

Closed without a decision. Program processing runs on the processing thread ([Where program processing runs](06-evaluation.md)), so caching there shortens the "Processing…" wait but not input latency, and this map ends at a spec about input latency. Caching bindings for the main thread's Edit-mode helpers stays with the cost of those helpers in the map's Not yet specified. Caching for the processing thread can come back as a separate effort about the "Processing…" wait.

## Re-measured

Re-measured on 2026-10-18 on the dev's machine (release) after the evaluation speed-ups (e54ec3c, 9d764f7), to check whether caching is still worth it. The old build is `e6e6508`, the commit the original instrumentation applies to, so both columns come from the same machine. The new build is `9d764f7` with that instrumentation ported. Both use the stand-in files and harnesses in [research/processing-timing](../research/processing-timing/README.md) (`instrumentation-9d764f7.diff` and `eval_timing-9d764f7.rs` for the new build).

| Warm reload | `e6e6508` | `9d764f7` | Speed-up |
|---|---|---|---|
| medium.tuun (26 programs) | 3840 ms | 526 ms | 7.3× |
| large.tuun (51 programs, 24 named) | 28.2 s | 12.8 s | 2.2× |

Where a warm reload's time goes at `9d764f7` (the "evaluation" row covers `eval::evaluate` and the second `on_beats` evaluation; the three rows indented under it break down the time spent building contexts, summed over both):

| | medium | large |
|---|---|---|
| Evaluation | 515 ms (98%) | 12.77 s (99.5%) |
| — earlier file bindings, evaluated again for each program | 207 ms (39%) | 11.53 s (90%) |
| — module exports (`use synth.sub`, `use synth.pm`, `open std`, …) | 224 ms (43%) | 590 ms (5%) |
| — the program's own expression | 32 ms (6%) | 309 ms (2%) |
| Second evaluation of `on_beats` programs (included in evaluation above) | 37 ms (7%) | 902 ms (7%) |
| Type check, optimize, parse | 8 ms | 58 ms |

- **Evaluation is still nearly all of a reload.** Counting only `eval::evaluate`, as the ticket's 93–95% did, it is 91% (medium) and 92% (large).
- **Module exports got about 15× cheaper:** 3390 → 224 ms over a medium reload; `use synth.pm` went from about 80 ms to 4.6 ms per use. Caching them now saves about 40% of a medium reload and 5% of a large one.
- **Earlier file bindings got only about 1.5× cheaper** (303 → 207 ms on medium, 17.8 → 11.5 s on large), and they stay quadratic in the number of named programs. They are now most of the cost on large.tuun. Caching the evaluated prefix of the file would remove it, and would also make the Edit-mode context fetch (spec step 3) close to free, since `Environment::program_context` evaluates the same prefix.
- **Processing one program** (leaving Edit mode, warm): 20–37 ms on medium (was 140–185 ms), and 11 ms, 315 ms and 620 ms for the first, middle and last program measured on large (was 145, 710 and 1160 ms).
- **`Environment::program_context`** (the context step 3 would fetch): at most 20 ms on medium and 308 ms on large.
- How much prefix caching matters for real files depends on how many named programs they have; large.tuun is a stand-in. Hypothesis (unverified): each earlier binding on large.tuun costs about 8 ms per evaluation because its named programs build large waveform trees.
