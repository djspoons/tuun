# Plan: the Edit-mode context prefetch (spec step 3)

Implements spec item 2 ([spec](spec.md), [Where completion and parameter hints run](tickets/11-completion.md)): completion and parameter hints (Cmd-/) stop evaluating on the main thread. Entering Edit mode asks the processing thread for the program's context, and Cmd-/ looks names up in the reply.

Each phase lands on its own and ends with the full verification suite and a pause for review.

## Where the context is evaluated today

Both helpers call `Environment::program_context(&state.programs, state.active_program_index)` on the main thread, through `ctx.environment`:
- `apply_complete` (`actions.rs`), when it starts a new cycle. Continuing a cycle reuses its stored candidates and evaluates nothing.
- `apply_parameter_hint` (`actions.rs`), on every Cmd-/ after `(`.

`program_context` prepends `open __prelude` to `ProgramSet::evaluation_bindings(index)` and calls `eval::evaluate_bindings`, resolving modules through the environment's module cache. It reads the set only through `evaluation_bindings`, which is already a fresh copy. Apart from a test in `programs.rs`, nothing else calls it.

Three facts found while mapping that the plan has to handle:

- **The failure mode is a copy of Edit mode.** Return sends `EvaluateAndLeaveEditMode { mode_on_failure: mode.clone() }` (`sdl2_input.rs`), and a failed evaluation restores that copy with `Effect::SetMode`. The leave-Edit job is queued behind the context job, so the context reply always lands between the copy and the restore. A context kept inside `Mode::Edit` would be overwritten by the copy's older "pending" state, and Cmd-/ would say "Preparing completions…" for the rest of the session. So the context lives on `AppState`, not in `Mode::Edit`.
- **A reload can't overlap an Edit session.** `Action::ReloadSource` does nothing outside Select mode, and `refused_by_processing` refuses `EnterEditMode` while a load is pending. So the ticket's "a reload fetches the context again" can never happen: every Edit session starts after any reload has been installed. Nothing in this plan refetches, and the spec and ticket get updated to say why.
- **Edit mode is entered in one place in production:** `Action::EnterEditMode` (`actions.rs`). Restoring `mode_on_failure` continues the same session. It is not a new one.

## Design

### The seam: a context from copied bindings

`Environment::program_context(&ProgramSet, usize)` becomes `Environment::context(bindings: Vec<SourceBinding<MarkId, Source>>)`, which prepends the prelude `open` and evaluates. The caller passes `set.evaluation_bindings(index)`. The processing thread can then evaluate a context from a copy, as `EvaluationInput` already lets it evaluate a program.

### The job

- `Job::Context(Vec<SourceBinding<MarkId, Source>>)` → `Reply::Context(Result<Vec<ContextEntry<MarkId, Source>>, String>)`. The error comes back as its message, which is all Cmd-/ shows ("Can't complete: …").
- Module values in the context carry the processing thread's module ids. Since phase 1 of the processing-thread plan, the module table is shared, so those ids mean the same thing on the main thread. `resolve_module_path` reads only the context.
- A compile-time `Send` assertion for the reply's context, next to the existing ones in `processing.rs`.

### Where the reply is kept

`AppState.edit_context: Option<EditContext>`:

```rust
pub struct EditContext {
    /// The Edit session this context belongs to.
    session: u64,
    /// `None` while the reply is on its way.
    reply: Option<Result<Arc<Vec<ContextEntry<MarkId, Source>>>, String>>,
}
```

and `AppState.next_edit_session: u64`. The `Arc` lets the parameter hint hold a looked-up value while `edit_text_op` borrows `state` mutably. Without it, the whole context would have to be cloned.

- **Entering Edit mode** (the reducer's `EnterEditMode`) takes the next session number, sets `edit_context` to pending for that session, and returns `Effect::FetchEditContext { program_index, session }`.
- **The runner** sends `Job::Context(state.programs.evaluation_bindings(program_index))` and pushes `PendingJob::Context { session }`. It does not set "Processing…": the fetch runs in the background, and the status line keeps the slider readout `EnterEditMode` just put there. If the send fails, it stores `Err("The processing thread has stopped")`.
- **`EffectRunner::finish`** gets a `Reply::Context` arm. It pops the front `PendingJob::Context` and stores the reply if `edit_context` is still waiting on the same session. Otherwise it drops the reply: the session ended, or a new one started.
- **Leaving Edit mode drops it.** Edit mode is left in several places, through `Effect::SetMode` and through direct writes in the reducer. Rather than clear the context at each one, `EffectRunner::dispatch` and `finish` each end by clearing `edit_context` when the mode isn't Edit. The reducer never reads the context outside Edit mode, so a context that outlives its session by part of one dispatch is never seen.
- **`is_processing(index)` stays program-only.** A pending context job doesn't block Edit-mode keystrokes, Return or anything else.

### Cmd-/

`apply_complete` (new cycle) and `apply_parameter_hint` read `state.edit_context` in place of calling `program_context`. They still run their own text checks first, so "Nothing to complete …" answers as it does now:
- pending: show "Preparing completions…" and change nothing. Nothing is remembered.
- `Err(message)`: show "Can't complete: {message}".
- `Ok(context)`: as today.

A `// TODO` at the lookup says a named default that depends on a slider shows the slider's value from when the session started (ticket decision 3).

`Context.environment` stays (ShowType uses it). Its doc stops naming `Action::Complete`.

### What it costs

- **Leaving Edit mode can wait behind the fetch.** Pressing Return before the context reply arrives queues the leave-Edit job behind it. In the worst case the "Processing…" wait grows by the context's evaluation time (69–317 ms release on `large.tuun`, per ticket 09). Input latency isn't affected, since the main thread is free throughout. Phase 3 measures how often this happens; see decision D.
- **Memory:** one context, a prefix of the file's evaluated bindings, while one program is in Edit mode. This is the ticket's decision 4.

## Phases

### Phase 1: the seam and the job

No behaviour change for the user: Cmd-/ still evaluates on the main thread, now through `context(set.evaluation_bindings(i))`.
- `Environment::context(bindings)` in place of `program_context`; update the two `actions.rs` callers and the `programs.rs` test.
- `Job::Context` / `Reply::Context` on the thread, plus the `Send` assertion.
- `reply_to` in the `effects.rs` tests handles the new job.
- Tests:
  - `processing`: the thread answers a context job with the bindings' names, in order with a program job sent before it. Uses stand-in sources.
  - `processing`: a context whose bindings don't evaluate comes back as `Err` carrying the message.

### Phase 2: prefetch on entering Edit mode

- `EditContext`, `AppState.edit_context` / `next_edit_session`, `Effect::FetchEditContext`, `PendingJob::Context`, the `finish` arm, the clear at the end of `dispatch` and `finish`.
- `apply_complete` and `apply_parameter_hint` read the stored context. The pending and error messages, and the slider TODO.
- Reducer tests that complete (`complete_*`, `parameter_hint_*`, `completion_cycle_*`) get a helper that fills `edit_context` synchronously: it evaluates `Environment::context` for the active program and stores `Ok`. Their assertions don't change.
- New tests:
  - Reducer: `EnterEditMode` sets a pending context and emits `FetchEditContext` with the new session.
  - Reducer: Cmd-/ with a pending context shows "Preparing completions…" and leaves text and cursor alone; with an `Err`, shows "Can't complete: …".
  - Runner: a context reply for the current session is stored, and the status line is untouched.
  - Runner: a reply for an ended session is dropped. Enter, leave and enter again, then answer both jobs: only the second is stored.
  - Runner: **a failed leave keeps the context.** Enter Edit, press Return, answer the context job and then the failing program job. The mode is Edit, with the context stored.
  - Runner: a successful leave and Escape both drop the context.
  - Runner: Edit-mode keystrokes aren't dropped while only a context job is pending.

### Phase 3: measure and record

- Re-measure with `--show-loop-timing` (release) on large.tuun: Cmd-/ (completion and parameter hint) on a program near the end of the file. Acceptance: neither shows in `keyboard input` beyond a few ms.
- Time enter-Edit → Return with no typing, on that program, to see the cost of leaving Edit while the fetch is pending (decision D).
- Record the numbers in ticket 11 and in the spec's Order of work. Update the spec's decision 2 and ticket 11 so neither says a reload refetches (a reload can't overlap an Edit session). Update the map.

## Decisions

- **A. The context lives on `AppState`, keyed by an Edit-session number**, not in `Mode::Edit`, because the leave-Edit failure path restores a copy of the mode that predates the reply (see "Where the context is evaluated today").
- **B. Drop on leaving Edit mode at the end of each dispatch and finish**, rather than at each place the mode changes, so a future way out of Edit mode can't leak a context.
- **C. No refetch on reload**, since reloads and Edit sessions can't overlap. The spec and ticket get updated, not the code.
- **D. Open: cancel a fetch whose session ended?** If Return comes before the reply, the leave waits behind a context nobody will use. Recommendation: don't build cancellation now. Measure it in phase 3, and add it only if the wait is noticeable. The cheapest form would be a shared atomic holding the current session, which the thread checks before starting a context job.
