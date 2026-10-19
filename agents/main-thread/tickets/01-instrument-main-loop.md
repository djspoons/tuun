# Instrument the main loop

Labels: wayfinder:task
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Add timing to the native UI so later decisions have numbers: how long each main-loop step takes (SDL events, Launchkey/MIDI events, the status wait, the recording tick, render, the Launchkey update) and how long each input event takes from arrival to its command being sent to the tracker, for the computer keyboard, Launchkey pads and Launchkey piano keys.

Decided while charting: keep the timings in the existing `Metric` (`src/lib/metric.rs`) and show them on screen like tracker load, behind a CLI flag. Done when the instrumented build compiles and passes the `CLAUDE.md` checks, and the flag and the numbers it shows are documented here.

## Resolution

Run the native UI with `--show-loop-timing`. Nine lines appear in the top-right corner, each as `<name>: <A> ms (10 s max <B>)`: A is the worst time in the current second, B the worst over the last ten seconds.

- **Steps of each main-loop pass:** `SDL events`, `Launchkey events`, `status wait` (includes the up-to-10 ms `recv_timeout`), `tick`, `render` (includes `present()`), `Launchkey update`. Each is the time the step took, including the handling of every event it dispatched.
- **Input latency,** from arrival to the end of the event's handling (after `dispatch_actions` returns):
  - `keyboard input`: every SDL key-down and text input (typed characters such as the digits that select a program arrive as text input). Measured from SDL's event timestamp with SDL's millisecond clock. SDL stamps an event when the main thread pumps it from the OS, so this excludes the time the key waited for the loop to come round. Treat it as a lower bound. Checked in SDL 2.32.2's source: `SDL_PushEvent` sets `timestamp = SDL_GetTicks()`, and on macOS `Cocoa_PumpEventsUntilDate` dequeues each `NSEvent` on the pumping thread and posts it from there.
  - `pad input`: Launchkey top and bottom pad presses in DAW mode.
  - `piano key input`: Launchkey note-ons.
  - Pad and piano-key events are stamped on the MIDI input thread when their message arrives, so these include the wait for the loop.
- "End of handling" means the main thread's work is done. With `--precompute` (the default), a played waveform still passes through the precompute thread before reaching the tracker; that hop is not included.

Code: `src/lib/loop_timing.rs` (new), `Metric::set_max` in `src/lib/metric.rs`, `Launchkey::events` now pairs each event with its arrival `Instant` (`src/lib/launchkey.rs`), and the wiring in `src/main.rs` and `src/lib/renderer.rs`. The `CLAUDE.md` checks pass (build, benches, tests, fmt, clippy, wasm tests).

## Ported to `main`

The instrumentation was first written on `claude/main-thread`, and the merge into `main` left it out. It was re-applied to `main` after [The 10 ms status wait](04-status-wait.md) and [Font loading per frame](05-font-loading.md) landed, with these differences:
- `status wait` is replaced by `idle`, the time blocked in `wait_event_timeout`, and `statuses`, the `try_recv` drain after waking.
- `render` and `Launchkey update` are recorded only on passes that render (at most 60 Hz).
- `render fonts` is gone, since fonts are no longer loaded per frame. The display draws 18 lines (the driver line plus 17), so `text textures` counts 18 more than a frame without the flag.
