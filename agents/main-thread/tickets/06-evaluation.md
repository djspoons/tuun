# Where program processing runs

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Should evaluating source (parsing, type checking and evaluating programs) move off the main thread, and if so, how do its results get back to the main loop? Measured in [Measure on the dev's machine](02-measure.md): reloading a large source file blocks the main thread for about 2.6 s, and leaving Edit mode with Return (which evaluates the program) for about 233 ms. Every input arriving meanwhile waits.

## Resolution

"Processing" here means parsing, type checking and evaluating a program; the user-visible status says "Processing…".

**Decisions**

1. **Move it.** Program processing moves off the main thread. Making it cheaper is a separate question: [Caching evaluated modules and bindings](07-evaluation-caching.md).
2. **Scope.** Every program evaluation: startup, source reload, leaving Edit mode, recording a take, sequencer step toggles, and undo/redo. Out of scope:
   - the Edit-mode helpers (completion, parameter hints, ShowType, the per-keystroke parse), which stay on the main thread until measured;
   - note-on, which belongs to [Where the note function runs](03-note-function.md);
   - batch mode (`--ui=false`), which stays synchronous.
3. **A processing thread.** One thread, started with the app, owns its own `Environment`. It receives jobs in order over an mpsc channel. Each job carries copies of everything it reads: the program's text, the text of the programs before it and their slider values (what `evaluation_bindings` reads today). It sends back waveforms, sequences, keys instruments and diagnostics. The main loop collects replies with `try_recv` each pass, as it does for tracker status and Launchkey events. The main thread keeps its own `Environment` for note-on and the Edit-mode helpers.
4. **First step: built-ins become `Send`.** `BuiltInImpl` (`src/lib/expr.rs`) changes from `Rc<dyn Fn …>` to `Arc<dyn Fn … + Send + Sync>`, so that `SourceExpr`, and with it an evaluated keys instrument, can cross threads.
   - A trial of that change left only two kinds of compile error: missing `Send + Sync` bounds on the mark type parameter `M` (`MarkId` satisfies them), and two tests whose `debug` sink is an `Rc<RefCell<…>>`. It was not taken to a clean build.
   - `Environment` stays non-`Send` (its caches are `RefCell`s); decision 3 does not need it to be `Send`.
   - `Environment::apply_note_function` needs no environment state, so once this lands an evaluated note function can be applied on any thread.
5. **Leaving Edit mode.** Return keeps the program in Edit mode with "Processing…" until its result arrives. On success the program leaves Edit mode, and plays if Cmd+Return asked for it; on failure the errors show, as now. Keystrokes for that program are dropped meanwhile. Other programs, pads, keys and sliders keep working.
6. **Reload.** Playback, pads and sliders keep working against the old programs. Actions that change source (editing, recording, sequencer toggles, undo/redo, another reload) are refused with "Processing…". The new program set is installed at once when all of its results are in.
7. **Startup** goes through the reload path: the window and input come up at once with "Processing…", and programs become playable when the set is installed. *Amended in [the processing-thread plan](../processing-thread-plan.md):* the source is still parsed on the main thread first, so a file that doesn't parse exits with its errors as before.
8. **Stale results.** Each program has a generation number that increases when its text changes. *Amended in [the processing-thread plan](../processing-thread-plan.md):* results apply in the order their jobs were sent, and one is dropped only when the program's text changed after the newest job for it was sent. Dropping every result but the newest would also drop the sequencer steps that quick toggles schedule.
9. **Write-back.** `UpdateSource` (`ProgramSet::splice`: re-parse and file write) stays on the main thread, in action order, next to the `source_stale` guard.

**Measurements behind these decisions**

Timed outside the UI on a 4-core Linux container, not the dev's machine, so only proportions transfer. The repo has no large source file, so two stand-ins were built:
- `medium.tuun`: 26 programs using `std`, `synth.sub`, `synth.pm`, `filters.rbj` and `env.minmax`.
- `large.tuun`: 51 programs, the first 24 of them named.

The README runs the UI as a debug build (`cargo run`). The harness and stand-in files are in [research/processing-timing](../research/processing-timing/README.md).

| Reload of medium.tuun, warm environment | release | debug |
|---|---|---|
| Total | 3881 ms | 8881 ms |
| `eval::evaluate` | 3596 ms | 8289 ms |
| second evaluation of `on_beats` programs | 265 ms | 523 ms |
| type check | 6 ms | 19 ms |
| optimize | 4 ms | 14 ms |

- **Evaluation** is about 93–95% of a reload.
  - Each program evaluates every module it opens or uses again, including their dependencies (`use synth.sub` about 60 ms and `use synth.pm` about 75 ms per program in release), and every named binding before it in the file. That makes a reload quadratic in the number of named programs: large.tuun took 32 s in release.
  - A callgrind profile put about 57% of instructions in malloc/free, under `eval::substitute` cloning expression trees.
- **Type checking** is cheap once its module cache is warm: under about 1 ms per program. A cold check costs 80–100 ms in release and 350–700 ms in debug. Startup evaluates every program, so the cache is warm by the time the user edits.
- **Write-back** (`splice`): the re-parse takes about 0.3–1 ms in release and 8–35 ms in debug; the write itself takes at most about 0.5 ms (up to 1.4 ms on large.tuun).
- **One program leaving Edit mode**, warm, medium.tuun: 140–190 ms in release and 290–370 ms in debug, nearly all of it `eval::evaluate`.

## Since the resolution

- **Measured after the move** (phase 5 of [the processing-thread plan](../processing-thread-plan.md), at 0b1c00b). These are release builds on the dev's machine, run with `--show-loop-timing` on the stand-in files. For this run the overlay gained a `replies` step, which times applying processing-thread replies on the main thread. Values are the overlay's 10 s max, in ms:

  | Scenario | File | SDL events | keyboard input | replies | render |
  |---|---|---|---|---|---|
  | Startup, while "Processing…" shows | medium | 0.2 | 2.0 | 0.0 | 7.6 |
  | Startup, while "Processing…" shows | large | 0.0 | 0.0 | 0.0 | not legible |
  | Reload, while processing | medium | 0.2 | 1.0 | 0.3 | 6.1 |
  | Reload, just after the set is installed | medium | 0.1 | 1.0 | 0.0 | 44.6 † |
  | Reload while playing a keys instrument, just after install | medium | 1.1 | 1.0 | 0.4 | 6.2 |
  | Reload, while processing | large | 1.4 | 2.0 | 0.0 | 42.7 † |
  | Reload, just after the set is installed | large | 0.2 | 0.0 | 1.2 | 37.1 † |
  | Leaving Edit mode (Enter, then Return) | medium | 8.4 † | 2.0 | 0.1 | 43.0 † |
  | Leaving Edit mode (Enter, then Return) | large | 1.3 | 2.0 | 0.1 | 6.3 |

  - **Before the move**, [Measure on the dev's machine](02-measure.md) recorded 2585 ms for a reload and 233 ms for leaving Edit mode, in both `SDL events` and `keyboard input`. Now handling a keystroke takes at most 2.0 ms (SDL's clock has 1 ms resolution). Applying a reply takes at most 1.2 ms, the cost of installing large.tuun's set. Notes played during a reload were handled within 1.0 ms.
  - **† Switching apps.** Hypothesis (unverified for each value): the marked values come from switching back to the Tuun window to take a screenshot, not from processing. Each is a `render other` spike (34.6–41.9 ms) or an `SDL events` pass. They match the "Idle, switching apps" column in [Measure on the dev's machine](02-measure.md), which recorded a 44.5 ms render and 12.7 ms of `SDL events`. The dev has seen switching to the Tuun window cause such spikes before. The two shots taken right after another shot both show one. The unmarked shots show at most a 7.6 ms render and 1.4 ms of `SDL events`.
  - **Startup:** `idle` peaked once at 312–313 ms on both files. It doesn't depend on the file, so processing doesn't cause it; the cause hasn't been investigated.
  - **Not measured:** sequencer toggles and undo/redo outside Edit mode (they need the Launchkey, which wasn't at hand), and pad latency.
- **Tracker load from a pad press**, before and after precompute moved to the processing thread. This was measured outside the UI with [tracker_load.rs](../research/processing-timing/tracker_load.rs) on medium.tuun (release, dev's machine). Each run plays one program with `Start::Now` into a tracker whose callbacks are paced at the 23.2 ms buffer period. "Before" is the same program run with precompute off. That sends the same tree a pre-phase-4 pad press sent: at a13729d, `Player::sender` routed `Start::Now` and `Start::At` around the precompute thread. Load is generate time over the buffer's duration, given as the range over three runs (two for `harm_line`):

  | Program | Not precomputed: max / p95 / mean | Precomputed: max / p95 / mean |
  |---|---|---|
  | A:6 | 0.053–0.054 / 0.046–0.047 / 0.040–0.042 | 0.002–0.004 / 0.001 / 0.0003 |
  | A:8 | 0.054–0.071 / 0.050–0.053 / 0.044–0.046 | 0.002–0.005 / 0.001–0.002 / 0.0003–0.0004 |
  | A:1 (`harm_line`, a sequence) | 0.114–0.116 / 0.039–0.047 / 0.038–0.043 | 0.002–0.003 / 0.001–0.002 / 0.0005–0.0006 |

  A pad press used to spend about 1–2.5 ms generating each buffer in the audio callback; it now spends a few microseconds. In return, processing one program on the thread took longer with precompute on: about 22 ms more for A:6 and `harm_line`, and about 100 ms more for A:8. That is more than the 75 ms maximum the plan measured for precompute alone; why wasn't investigated.

- **Command order on the way to the tracker.** Before phase 4 of [the processing-thread plan](../processing-thread-plan.md), a next-measure play went through the precompute thread, while `RemovePending` and stops went straight to the tracker. Hypothesis (never verified): a `RemovePending` or stop sent right after a next-measure play could reach the tracker first, find nothing to remove, and let the play start anyway. Phase 4 gave the `Player` one sender, so the commands it sends reach the tracker in the order it sends them. The old path is gone, so the hypothesis can no longer be tested; it is recorded here only so that the "too late" window in `agents/tracker-timing/audit.md` isn't blamed on it without evidence.
