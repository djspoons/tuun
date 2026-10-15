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
