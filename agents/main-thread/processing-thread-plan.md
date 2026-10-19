# Plan: the processing thread (spec step 2)

Implements spec items 1 and 3 ([spec](spec.md), [Where program processing runs](tickets/06-evaluation.md)): parsing, type checking and evaluating programs move off the main thread. Item 2 (the Edit-mode context prefetch) is spec step 3 and is not part of this plan, but the job queue below is built so it can join as another job kind.

Each phase lands on its own and ends with the full verification suite and a pause for review.

## Where processing runs today

All of it is synchronous on the main thread:

| Trigger | Code | Effects after the evaluation |
|---|---|---|
| Startup | `main.rs`, a loop emitting `Effect::EvaluateProgram` per program | the "Errors in …" message |
| Reload (Cmd+R) | `Effect::ReloadSource` in `effects.rs`: `reload_from_disk` (read + parse), stop all, swap the set, `evaluate_and_record` per program | "Reloaded … — failed: …" |
| Leaving Edit mode | `Action::EvaluateAndLeaveEditMode` | `UpdateSource`, then `PlayProgram` if Cmd+Return |
| A finished take | `apply_take_finish` (`actions.rs`) | `UpdateSource`, then `PlayProgram { start: At(end) }` |
| Sequencer toggle | `apply_toggle_sequencer_step` | `UpdateSource`, `ShowMessage`, `PlaySequencerStep` / `RemovePendingSequencerStep` |
| Undo/redo outside Edit | `apply_committed_history_restore` | `UpdateSource`, `PlaySequencerStep`s / `RemovePendingSequencerStep`s |

All evaluation funnels through `ProgramSet::evaluate_and_record(&Environment, index)`, which calls `Environment::check_program(set, i)` and `Environment::evaluate_program(set, i)`. Both read the set only through `evaluation_bindings(i)`, the program's text and kind, and (in `diagnose`) the file source for file-located errors.

Two facts found while mapping that the plan has to handle:

- **Effects that follow an evaluation read its result.** `PlayProgram` and `PlaySequencerStep` read `program.evaluated()`, so once evaluation is asynchronous they have to wait for the reply. `UpdateSource` doesn't (it splices text), and it stays in action order on the main thread (spec item 7).
- **Module ids are per environment.** `Environment::record_module_info` numbers modules in the order *that* environment loads them, and spans carry `Source::Module(id)`. A keys instrument evaluated on the processing thread carries the thread's ids; a note-on error from it is diagnosed on the main thread (`Effect::PlayNoteOn` → `self.environment.diagnose`) against the main environment's table, which numbers modules differently. That error names the wrong module file: verified in phase 1 by `environment::tests::module_errors_diagnose_across_environments`, which, run with two independent environments that loaded `first` and `second` in opposite orders, named `first.tuun` for an error in `second.tuun`. The shared table fixes it.

## Design

### The seam: a copy of what evaluation reads

`programs::EvaluationInput` holds what evaluating program *i* reads: its text, kind, `evaluation_bindings(i)` (already a fresh copy today), and the file source for file-located diagnostics. `ProgramSet::evaluation_input(i)` builds it.

`Environment::check_program`, `evaluate_program` and `diagnose` take the input (or the text and source it holds) instead of `(&ProgramSet, index)`. `evaluate_and_record` splits into:
- `processing::process(&Environment, &EvaluationInput) -> Outcome`: today's body (empty-text shortcut, checker gate, evaluate), returning `Result<(Option<Evaluated>, warnings), diagnostics>`.
- `ProgramSet::record(i, Outcome)`: stores the evaluated value.

and `processing::load(&Environment, source, input_path) -> Result<Loaded, String>` holds today's reload body after the file read: `ProgramSet::from_source`, then `process` + `record` for each program, returning the set, the parse warning and the failed programs.

Batch mode (`--ui=false`) calls these synchronously, as it does `evaluate_and_record` today.

### The thread

`processing::spawn(sample_rate, tempo, library_root, check_mode, module_ids, wake)` starts one thread that builds its own `Environment` (it isn't `Send`, so it is built on the thread) and returns a `Sender<Job>` and a `Receiver<Reply>`:

- `Job::Program(EvaluationInput)` → `Reply::Program(Outcome)`
- `Job::Load { source, input_path }` → `Reply::Loaded(Result<Loaded, String>)`

It handles jobs in order, sends each reply, then calls `wake` (`main_loop_waker`, as the Launchkey does). The main loop drains replies with `try_recv` right after the tracker statuses, before input, so an input in the same pass sees the result.

**Module ids** become a table both environments share (`environment::ModuleTable`, an `Arc<Mutex<Vec<ModuleInfo>>>`; `Environment::with_module_table` takes one and `Environment::module_table` hands out its own), so an id means the same file on both threads. Ids are already looked up by path, so sharing changes nothing else.

### Pending jobs and their continuations

`AppState.processing` is a queue of `PendingJob`s in submission order (replies come back in the same order, so each reply matches the front):

- `Program { index, generation, on_success: Vec<Effect>, on_failure: Vec<Effect>, message }`
- `Load { startup: bool }`

`Effect::EvaluateProgram` carries `on_success` and `on_failure`: the effects that run once its result is recorded, chosen by whether the evaluation succeeded. A mode switch that waits on the result is `Effect::SetMode` in one of them. At each of the four call sites above, everything after the evaluation except `UpdateSource` moves into them: the effects that read the result (plays) and the toggle's message go in `on_success`, and those that read nothing (sequencer step removals) go in both. The runner's `EvaluateProgram` sends the job and pushes the `PendingJob`. A new `EffectRunner::finish(state, world, reply)` pops the front, records the result, sets the message as `EvaluateProgram` does today, and runs `on_success` or `on_failure`. On failure it then hands the diagnostics to Edit mode if that is the mode `on_failure` left. (Phase 2 had one `then` list, which ran either way, beside two optional modes. It was split after phase 3. Under it, a sequencer toggle whose evaluation failed replaced the error with its "beat … on" message.)

The reducer reads `state.processing` to refuse things (below), and the runner sets "Processing…" when it sends a job. The `PendingJob::Program` keeps the message that "Processing…" replaced, and `finish` puts it back if the status line still says "Processing…", before applying the result, so a take's "Recorded …" or an undo's "Undid …" posted before the evaluation survives it (added in phase 2).

### Generations

`Program` gets a `generation` that `set_text` bumps (`resync_text` must not: splice calls it with semantically identical text right after a leave-Edit job is sent). A `PendingJob::Program` records the generation it was sent with. See decision A for when a result is dropped. A splice that deletes a program replaces it with a fresh padding program; that program keeps the old generation, or the reply to the leave-Edit job that deleted it would be dropped and leave the editor stuck in Edit mode (added in phase 2).

`set_text` no longer drops the evaluated result: a program keeps its last result until the next one is recorded, so a pad press between an edit (a sequencer toggle, an undo, a take) and its reply plays the previous version instead of nothing. A failed evaluation still drops the result. The sequencer pad grid reads its steps from the text only, since the evaluated sequence lags behind it while an edit is being evaluated (both changed in phase 2).

### Leaving Edit mode

Return sends the job and stays in Edit mode. While a job for the active program is pending in Edit mode, the Edit-mode keyboard actions (text and cursor edits, Complete, Undo/Redo, Return, Escape) are dropped. ShowType stays allowed (it reads only). Pads and keys keep working, subject to the ownership refusals that already exist. The reply switches to Select (`on_success`) or to the failure mode (`on_failure`) as today; with Cmd+Return, the `PlayProgram` in `on_success` queues from the next measure after the reply.

### Reload and startup

- **Reload** reads the file on the main thread (a fast `read_to_string`) and sends `Job::Load`. Nothing is stopped yet: playback, pads and sliders keep working against the old set. On the reply, an error leaves everything as is (as today). Success stops all voices, clears pending playback and keys, installs the set, resets the slider baselines, and posts the message. Today's stop-before-evaluate moves to install time.
- **While a load is pending**, the actions that change source are refused with "Processing…": EnterEditMode, ToggleSequencerStep, arming a take, Undo/Redo, another ReloadSource. `UpdateSource` is skipped, since the install replaces the set (it also runs on pad presses through `StartProgramVoice`, which should keep playing). Slider and level moves made meanwhile are lost at install, as they would be in today's reload.
- **No stale-set replies.** Program jobs sent before the reload finish before the load reply (one thread, in order), and none are sent while it's pending, so a program reply never lands on the new set. A test pins this down. A program reply that lands while the load is pending leaves "Processing…" on the status line instead of restoring the message it replaced (added in phase 3).
- **Startup** still parses on the main thread so that a parse error exits with printed errors as today. The window comes up showing the unevaluated programs with "Processing…", and a `Job::Load { startup: true }` with the same source runs the reload path. On install, `startup` picks today's "Errors in …" message instead of "Reloaded …". The source is parsed twice at startup, which costs a few ms.

### Precompute at evaluation

Today a separate precompute thread sits in front of the tracker as the Player's `precompute_sender` route. It runs `Generator::precompute` on every `Start::NextMeasure` play (and the Beats waveforms), each time it is played. Immediate plays (`Start::Now`, `Start::At`, voices, sequencer steps scheduled live) take the `fast_sender` route and skip precompute.

Measured on the dev's machine (release, medium.tuun, a throwaway harness): precompute takes 0.01–75 ms per program (265 ms over all 25 waveform programs), against 90–350 ms to evaluate one (4.6 s in all). A sequence precomputes its step once per step: harm_line's step takes 17.7 ms, four times per play.

The processing thread precomputes once instead: `processing::process` runs it on the waveform a program plays (`Evaluated::Waveform`'s waveform, or `Evaluated::Sequence`'s `step_waveform`), and the cached result is what every play sends. This works because precompute never folds a `Marked` node into a `Fixed` one (`precompute_internal` treats `Marked` as dynamic), so the slider substitution and the level/terminator/`TopLevel` wrap that happen at play time act on the same nodes either way. In the harness, precomputing the evaluated waveform first and then substituting and wrapping gave the same tree as precomputing the played form, for all 25 waveform programs (compared by hashing their `Debug` output, with sliders at their file values).

What it changes:
- Every route plays precomputed waveforms, including pad presses, voices, the take handoff's `Start::At` and live sequencer steps, which today generate their whole tree in the audio callback.
- A play costs no precompute: playing a program again, or each step of a sequence, reuses the cached result.
- `Start::NextMeasure` plays reach the tracker at once. Today they wait behind precompute, which can push them past the measure boundary they were scheduled for (the "too late" window in `agents/tracker-timing/audit.md`). Hypothesis (unverified): a `RemovePending` or stop sent on the fast route right after a next-measure play can reach the tracker first today, so it finds nothing to remove. One route ends that ordering question either way.
- The precompute thread goes, and the Player keeps one sender. `--precompute` becomes a setting of the processing thread.
- `start_beats` precomputes its two Beats waveforms itself before sending them. It runs once at startup, before the window opens.
- Batch mode doesn't precompute, as today.

What it costs:
- Every program is precomputed when it is evaluated, played or not: on medium.tuun that adds about 6% to a reload, all of it off the main thread.
- A program's samples stay in memory for as long as its evaluation is cached, not only while it plays. `generate_fixed` caps one waveform at 2^28 samples (1 GiB); a typical one is a few seconds (under 1 MB per second at 44.1 kHz).

### What stays on the main thread

Its own `Environment`, used for note-on, PrintEvaluatedProgram, completion and parameter hints (until spec step 3), ShowType and the per-keystroke parse. `UpdateSource` (the write-back) runs in action order next to the `source_stale` guard.

One consequence: the main environment's type cache is no longer warmed by startup, so the **first ShowType** after startup or after a library edit pays a cold check (80–100 ms release, 350–700 ms debug on the container, per ticket 06). Recorded in the spec's gaps; it goes away if ShowType moves to the thread later.

## Phases

### Phase 1: the seam, still synchronous

No behaviour change.
- `EvaluationInput`, `ProgramSet::evaluation_input`, `processing::process`, `processing::load`, `ProgramSet::record`. The Environment entry points take the input.
- Startup, `Effect::EvaluateProgram`, `Effect::ReloadSource` and batch mode call them synchronously.
- The shared module-id table. First a test that shows the cross-environment misattribution (a keys instrument evaluated in one environment whose note-on error is inside a module, diagnosed in another), then the fix.
- Compile-time `Send` assertions for `EvaluationInput`, `Outcome` and `Loaded` (`ProgramSet` inside it), next to the existing one in `expr.rs`.
- Tests for `process` and `load` with stand-in sources, replacing the `evaluate_and_record` tests.

### Phase 2: the thread and per-program jobs

- `processing::spawn`, `Job`, `Reply`, the main-loop drain, and the waker.
- `Program.generation`, `AppState.processing`, `Effect::EvaluateProgram { then }`, `EffectRunner::finish`.
- The four call sites move their trailing effects into `then` (later split into `on_success` / `on_failure`).
- Leaving Edit mode pends as described; the Edit-mode keyboard drop.
- Reload and startup still run `processing::load` synchronously on the main thread in this phase. Startup installs the set `load` builds, so the source is already parsed twice.
- Tests:
  - Runner tests hold the job receiver and play the thread themselves (call `process`, then `finish`), so they're deterministic.
  - Reducer tests for the Edit-mode drop and the `then` lists (now `on_success` / `on_failure`).
  - One test drives the real thread with a small source and waits on the reply with a timeout.

### Phase 3: reload and startup through the thread

- `Job::Load`, install on reply, the load-pending refusals, `UpdateSource` skipped while loading, startup through it.
- Tests: refusals while loading; a program job sent before the reload is applied to the old set and the load installs after it; a failed load leaves state untouched.

### Phase 4: precompute at evaluation

- `processing::process` precomputes `Evaluated::Waveform`'s waveform and step waveform when the processing thread's `precompute` setting is on (`--precompute`, default on); `processing::spawn` takes the setting. Batch mode passes it off. (As built: `Evaluated` became three arms, `Waveform`, `Sequence` and `KeysInstrument`, so a sequenceable program no longer keeps its whole pattern, which nothing played, and precompute maps each arm to the one waveform it plays. `Player::play_program` dispatches on the arm, and batch mode now plays a sequence decomposed, as the UI does. Batch mode waits on `Terminator` marks rather than `TopLevel` ones, since a sequence's anchor can end before its last step. `Environment::sample_rate` was added so `process` and `start_beats` can build a generator.)
- Delete the precompute thread from `main.rs`. `Player` keeps one sender; `Player::sender` and the routing in the `Start` docs go. `start_beats` precomputes the Beats waveforms before sending them.
- Update the tests that build a `Player` with two channels.
- Tests:
  - An integration test (it uses the real generator and player wrap): a waveform with a slider mark around a precomputable finite part, precomputed and then substituted and wrapped, equals the played form precomputed. A slider moved after precompute still reaches the tracker as a `Marked` node.
  - `process` with precompute on returns `Fixed` nodes for a finite program and leaves an infinite one alone.

### Phase 5: measure and record

- Re-measure with `--show-loop-timing` (release) on medium.tuun and large.tuun: reload, leaving Edit mode, a sequencer toggle, startup. Acceptance: none of them shows in `SDL events` or `keyboard input` beyond a few ms.
- Note the tracker load while a precompute-heavy program (medium.tuun's A:6 or A:8) plays from a pad, before and after phase 4.
- Record the numbers in ticket 06 and the spec's Order of work. Update the map with the measurements. (The ShowType cold-cache gap is already in the spec, and the map already reflects phase 4.)
- (As measured, in [Where program processing runs](tickets/06-evaluation.md): the acceptance holds for reload, leaving Edit mode and startup on both files. The sequencer toggle wasn't measured, because it needs the Launchkey. Tracker load was measured outside the UI with `research/processing-timing/tracker_load.rs`, comparing precompute off and on, rather than in the app before and after phase 4. A pad press can't be made without the Launchkey, and with precompute off the harness sends the tree a pre-phase-4 pad press sent.)

## Decisions

- **A. Stale results:** results apply in submission order; one is dropped only when the program's text changed after the newest job for it was sent. EnterEditMode is refused for a program with a pending job. The spec's literal rule, which drops every result but the newest, was rejected because quick sequencer toggles would lose the steps they schedule.
- **B. Startup:** the source is parsed on the main thread (a bad file still exits with its errors), then evaluated through the load job. Starting from an empty set was rejected because an empty set with a backing path can splice over the user's file.
- **C. Precompute:** precompute moves from its own thread to the processing thread, once per evaluation, and the cached result is what every play sends (phase 4).
