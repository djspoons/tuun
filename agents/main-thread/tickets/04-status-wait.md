# The 10 ms status wait

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

Should the main loop stop blocking for up to 10 ms (`recv_timeout`) while it waits for a tracker status, and if so, what paces the loop instead? The Scope views map (`agents/scan-viz/` on `claude/scan-viz`) changes how samples reach the UI but leaves this status drain unchanged.

## Resolution

Remove the wait. The tracker sends one status per audio callback, every ~23 ms at the defaults (1024 samples at 44.1 kHz), so in most passes the 10 ms `recv_timeout` simply times out: it is the loop's sleep, and every input arriving during it waits up to 10 ms before dispatch. Nothing else paces the loop; the renderer does not request vsync.

- **One wake source.** The loop blocks in SDL's `wait_event_timeout`. These wake it:
  - SDL input.
  - Launchkey events: midir's callback posts a payload-free wake event through `EventSender`; the events stay on their `mpsc` channel with their arrival times.
  - Processing-thread results ([Where program processing runs](06-evaluation.md)), the same way.
  - Tracker statuses do *not* wake the loop. Each pass drains them with `try_recv`, which keeps the audio thread off SDL's event queue.
- **Render at most 60 Hz.** Each wake dispatches everything queued; the loop renders only once the frame deadline (about 16.7 ms after the last render) has passed. The timeout is the earliest of the next render, the next tick, and the next buffer refresh (`SendCurrentBuffer`, every 200 ms). An input then waits only for a render already in progress (about 5 ms at most), never for a timeout.
- **Ticks** join the wake deadlines and fire at their deadline, rounded up to the next millisecond (the resolution of `wait_event_timeout`). This settles the TODO after the status drain in `src/main.rs`.
- **Acceptance.** With `--show-loop-timing`, the "status wait" step becomes "idle" (time blocked with nothing to do). Re-run the "Several programs playing" and "Keys instrument" columns of [Measure on the dev's machine](02-measure.md): maximum keyboard latency of about 6 ms (one render in flight plus dispatch). Check pad and piano-key latency against the same number once the Launchkey is available.
- **Input thread.** Not needed for this: an input waits only behind a render in progress. Whether to revisit it folds into the render-cost fog on the map.

## Implementation

Code: `main_loop_waker` and the main loop in `src/main.rs`; `EventSink` in `src/lib/launchkey.rs`, which sends each Launchkey event and then wakes the loop; `AppState::next_take_tick` in `src/lib/actions.rs`. The wake event's type comes from `register_event`, and the loop skips it after waking.

Choices the resolution left open:
- Each pass drains statuses right after waking, before dispatching, so inputs dispatch against the newest status.
- The Launchkey LED update runs with the render, at most 60 Hz, not on every pass.
- The frame interval is `renderer::FRAME_BUDGET`, counted from the start of the last render.

Checked: the `CLAUDE.md` checks pass (build, benches, tests, fmt, clippy, wasm tests). The app starts and renders with `test1.tuun`. Steady-state CPU in a debug build with no Launchkey fell from about 24% to 17% (10 s samples).

Not checked: pad and piano-key wakes haven't been run against hardware.

## Measurements after the change

Release build on the dev's machine, with `--show-loop-timing` ported to `main` ([Instrument the main loop](01-instrument-main-loop.md)). Each value is `current second / 10 s max`, in ms. Before-change numbers are from [Measure on the dev's machine](02-measure.md).

| Line | Idle | Several programs playing, just after typing | Edit mode, long program |
|---|---|---|---|
| idle | 15.1 / 15.1 | 13.2 / 15.1 | 15.1 / 15.4 |
| statuses | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| SDL events | 0.0 / 0.1 | 0.0 / 0.5 | 0.0 / 0.4 |
| tick | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| render | 5.6 / 8.4 | 5.6 / 6.8 | 5.9 / 6.1 |
| keyboard input | 0.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |

- **The status wait is gone.** Where a pass used to spend 11 ms in `recv_timeout`, the loop now sleeps in `idle` for most of each 16.7 ms frame, and SDL events dispatch in at most 0.5 ms (9.2 ms before, playing).
- **Keyboard latency reads at most 1.0 ms,** under the 6 ms target. But SDL stamps a key when the main thread pumps it, so this line can't show a key waiting behind a render ([Instrument the main loop](01-instrument-main-loop.md)). The true worst case is bounded by one render plus dispatch, about 7 ms with the 6.8 ms render maximum above. That is close to the target but unmeasured.
- **Leaving Edit mode is still long-running,** as expected until program processing moves off the main thread (spec step 2). Just after leaving Edit mode on a long program: `SDL events` 0.1 / 342.8, `keyboard input` 0.0 / 343.0, `render` 5.9 / 6.6, `text textures` 31 / 35. That is more than the 233 ms in [Measure on the dev's machine](02-measure.md). Hypothesis (unverified): the program evaluated here is costlier, or sits later in its file, than the one measured then. The status wait removal doesn't touch this path.
- **Keys instrument:** not re-measured. The change doesn't touch note-on cost, and the typing run covers keyboard dispatch under the new loop.
