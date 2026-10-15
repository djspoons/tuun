# Main-thread work and input latency

Labels: wayfinder:map

## Destination

A spec, in this directory, that ranks the work on the native UI's main thread by its effect on input latency, with a decision for each item: keep it there, cache it, or move it to another thread.

## Notes

- Input latency: the time from an input event (computer keyboard in Keys mode, Launchkey pad, Launchkey piano key) to its command reaching the tracker. All three paths pass through the main loop in `src/main.rs`.
- Long-running: slower than one frame, about 16 ms.
- Each main-loop pass, in order: poll and dispatch SDL events; drain and dispatch Launchkey/MIDI events; wait up to 10 ms for a tracker status (`recv_timeout`); the recording tick; render (`Renderer::render`, which loads fonts each frame); update the Launchkey.
- A note-on evaluates the keys instrument's note function on the main thread (`environment.apply_note_function` in `src/lib/effects.rs`).
- Measurements need the native UI with a display and audio device, so they are run by the dev on their own machine.
- Follow `CLAUDE.md`.
- Tickets and their workflow: see [README.md](README.md).

## Decisions so far

- [Instrument the main loop](tickets/01-instrument-main-loop.md): `--show-loop-timing` shows each step's time and each input's latency on screen; keyboard latency is a lower bound.
- [Measure on the dev's machine](tickets/02-measure.md): idle passes spend about 11 ms in the status wait and 5-7 ms rendering; source reload blocks about 2.6 s and leaving Edit mode about 233 ms; a note-on takes at most about 1 ms; pad and Launchkey key latency not yet measured.
## Not yet specified

- Render cost beyond font loading, and whether `present()` waits for vsync.
- Whether input should be handled on its own thread rather than in the main loop.

## Out of scope

- The audio thread, the precompute thread, and the web/wasm UI.
- Making the moves themselves: this map ends at the spec.
