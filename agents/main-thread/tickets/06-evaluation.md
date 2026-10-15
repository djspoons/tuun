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
7. **Startup** goes through the reload path: the window and input come up at once with "Processing…", and programs become playable when the set is installed.
8. **Stale results.** Each program has a generation number that increases when its text changes. A result is applied only if its generation still matches; otherwise it is dropped.
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
