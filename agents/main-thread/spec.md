# Main-thread work and input latency: spec

The work on the native UI's main thread, ranked by how long it can keep an input waiting, with a decision for each item. Each decision's reasoning and measurements live in the linked ticket; this spec gives only the gist.

**Input latency** is the time from an input event (computer keyboard in Keys mode, Launchkey pad, Launchkey piano key) to its command reaching the tracker. Every input passes through the main loop in `src/main.rs`, so any work on the main thread delays the inputs that arrive while it runs. **Long-running** means slower than one 60 Hz frame, about 16.7 ms.

Out of scope: the audio thread, the precompute thread, the web/wasm UI, and making the changes themselves.

## Ranking

Ranked by the worst time measured on the dev's machine with `--show-loop-timing`, comparing release builds where both builds were measured. Items measured only on a 4-core Linux container are ranked by those numbers and marked *(container)*; only their proportions transfer to the dev's machine.

| # | Work | Worst measured | Decision |
|---|---|---|---|
| 1 | Program processing on source reload and startup | 2585 ms release | **Move** to the processing thread |
| 2 | Completion and parameter hints (Cmd-/) | 317 ms release, 920 ms debug | **Move** the evaluation to the processing thread; **cache** its result for the Edit session |
| 3 | Program processing on leaving Edit mode, recording, sequencer toggles, undo/redo | 233 ms release | **Move** to the processing thread |
| 4 | Rendering text in Edit mode | 22.8 ms render, 20.9 ms of it text | **Keep**; draw the active line in runs |
| 5 | The status wait (`recv_timeout`) | 10 ms per pass | **Remove**; the loop waits on SDL events |
| 6 | Note-on (the keys instrument's note function) | about 1 ms; 8–10 ms for a four-note chord in debug *(container)* | **Keep** |
| 7 | Write-back of edited source (`ProgramSet::splice`) | 8–35 ms debug, 0.3–1 ms release *(container)* | **Keep** |
| 8 | Loading fonts each frame | 0.8–1.5 ms; estimated 1–4 ms with the glyph cache | **Cache**: load once |
| 9 | ShowType (Ctrl-H) | 2.5 ms debug | **Keep** |
| 10 | Rendering an ordinary frame (other than fonts and the Edit-mode line) | 5–6 ms before the changes above | **Keep** |
| 11 | Re-parse after each Edit-mode keystroke | 0.5 ms | **Keep** |
| 12 | Launchkey events and update, recording tick | 0.0 ms | **Keep** |

Note-on and the write-back rank above their usual cost because their worst cases (chords in debug, re-parse in debug) are close to or above the budget they're held to.

## Decisions

### 1–3. Program processing moves to a processing thread

From [Where program processing runs](tickets/06-evaluation.md):
- Parsing, type checking and evaluating programs move to one thread, started with the app, that owns its own `Environment`. Jobs reach it in order over an mpsc channel and carry copies of everything they read. Results come back to the main loop and wake it.
- **First step:** built-ins (`BuiltInImpl` in `src/lib/expr.rs`) become `Arc<dyn Fn … + Send + Sync>`, so evaluated expressions can cross threads.
- While a job is pending the user sees "Processing…" and input keeps working:
  - Leaving Edit mode keeps the program in Edit mode until its result arrives, and drops keystrokes for that program meanwhile.
  - During a reload, playback, pads and sliders keep working against the old programs, and actions that change source are refused.
  - Startup goes through the reload path.
- Each program has a generation number, and a result whose generation is out of date is dropped.
- The write-back stays on the main thread (item 7).
- Batch mode (`--ui=false`) stays synchronous.
- Making processing itself faster is out of scope: it shortens the "Processing…" wait, not input latency ([Caching evaluated modules and bindings](tickets/07-evaluation-caching.md)).

### 2. Completion and parameter hints use a prefetched context

From [Where completion and parameter hints run](tickets/11-completion.md):
- Both helpers spend their time evaluating the program's context: the bindings before it in the file, plus slider values. The program's own text doesn't affect that context, so it stays valid for the whole Edit session.
- Entering Edit mode queues a job on the processing thread that evaluates the program's context. The main thread keeps the result until it leaves Edit mode, and Cmd-/ looks names up in it.
- Cmd-/ before the context arrives shows "Preparing completions…" and does nothing.
- A reload fetches the context again. Slider moves don't: a parameter hint's default may show an older slider value (a TODO in the code).
- Only the context of the program in Edit mode is kept.

### 4. Edit-mode text is drawn in runs

From [Caching text textures](tickets/10-text-textures.md) and [Measure render cost](tickets/08-measure-render.md):
- The active Edit-mode line is drawn one texture per character today, about 112 textures for a long program.
- Drawing it in runs of one colour, with the cursor placed by `font.size_of`, brings a frame down to about 13–15 textures.
- There is no text-texture cache across frames.

### 5. The status wait is removed

From [The 10 ms status wait](tickets/04-status-wait.md):
- The loop blocks in SDL's `wait_event_timeout`. SDL input, Launchkey events and processing-thread results wake it; tracker statuses don't, and are drained with `try_recv` each pass.
- The loop renders at most every 16.7 ms. Its timeout is the earliest of the next render, the next tick and the next buffer refresh.
- An input then waits only for a render already in progress.

### 6. Note-on stays on the main thread

From [Where the note function runs](tickets/03-note-function.md):
- The budget is about 5 ms per loop pass, counting every note-on in that pass.
- If note-on ever moves, it never shares the processing thread.
- Cheaper note-ons are code TODOs in `Environment::apply_note_function` and `PlayNoteOn`.

### 7–12. The rest stays

- **Write-back** stays on the main thread, in action order, next to the `source_stale` guard ([Where program processing runs](tickets/06-evaluation.md)).
- **Fonts** are loaded once and held by `Renderer<'ttf>` ([Font loading per frame](tickets/05-font-loading.md)).
- **ShowType and the per-keystroke parse** stay on the main thread; both are at most 2.5 ms ([Measure the Edit-mode helpers](tickets/09-measure-edit-helpers.md)).
- **Everything else** (the rest of rendering, Launchkey events and update, the tick) measured at most a few ms and stays.

## Order of work

Each step can land on its own:

1. Load fonts once (item 8) and draw the Edit-mode line in runs (item 4). Neither depends on anything else.
2. Make built-ins `Send` (the first step of items 1–3).
3. Add the processing thread and move program processing onto it (items 1 and 3).
4. Prefetch the Edit-mode context on it (item 2).
5. Remove the status wait (item 5). It can come before step 3; processing-thread results join its wake sources when the thread exists.

## Acceptance

Re-measure with `--show-loop-timing` on the dev's machine:
- **Keyboard latency** has a maximum of about 6 ms in the "Several programs playing" and "Keys instrument" scenarios of [Measure on the dev's machine](tickets/02-measure.md).
- **Source reload, leaving Edit mode and Cmd-/** no longer show in `SDL events` or `keyboard input` beyond a few ms.
- **An Edit-mode render** of a long program takes under 16.7 ms.
- **Pad and piano-key latency** match the keyboard's once the Launchkey is available.

## Gaps and open questions

- **Launchkey not measured.** Pad and piano-key latency have no numbers yet; the hardware wasn't at hand.
- **Note-on under heavier instruments.** It's open whether note-on stays within its budget as keys instruments get heavier, and for chords in debug builds.
- **A cross-frame text cache** comes back only if later measurements show ordinary frames over budget.
