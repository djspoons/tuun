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
  - `keyboard input`: every SDL key-down. Measured from SDL's event timestamp with SDL's millisecond clock. Hypothesis (unverified): SDL stamps an event when it pumps it from the OS, at the start of the `SDL events` step, so this excludes the time the key waited for the loop to come round. Treat it as a lower bound.
  - `pad input`: Launchkey top and bottom pad presses in DAW mode.
  - `piano key input`: Launchkey note-ons.
  - Pad and piano-key events are stamped on the MIDI input thread when their message arrives, so these include the wait for the loop.
- "End of handling" means the main thread's work is done. With `--precompute` (the default), a played waveform still passes through the precompute thread before reaching the tracker; that hop is not included.

Code: `src/lib/loop_timing.rs` (new), `Metric::set_max` in `src/lib/metric.rs`, `Launchkey::events` now pairs each event with its arrival `Instant` (`src/lib/launchkey.rs`), and the wiring in `src/main.rs` and `src/lib/renderer.rs`. The `CLAUDE.md` checks pass (build, benches, tests, fmt, clippy, wasm tests).

