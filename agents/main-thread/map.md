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
- [Where program processing runs](tickets/06-evaluation.md): parsing, type checking and evaluating programs move to one processing thread that owns its own `Environment`. First, built-ins become `Send`. Leaving Edit mode, reloading and startup show "Processing…" while input keeps working. The write-back stays on the main thread. Evaluation, not type checking or I/O, is about 93–95% of the cost.
- [Where the note function runs](tickets/03-note-function.md): it stays on the main thread. The budget is about 5 ms per loop pass, chords included. A note-on costs about 0.1–0.9 ms in release and 0.3–2.4 ms in debug, and it never moves to the processing thread. Cheaper note-ons are code TODOs.

## Not yet specified

- Render cost beyond font loading, and whether `present()` waits for vsync.
- Whether input should be handled on its own thread rather than in the main loop.
- Whether note-on stays within budget as keys instruments get heavier, and for chords in debug builds (a four-note flute chord is about 8–10 ms).
- Cost of the Edit-mode helpers (completion and parameter hints evaluate all preceding bindings; ShowType; the per-keystroke parse).

## Out of scope

- The audio thread, the precompute thread, and the web/wasm UI.
- Making the moves themselves: this map ends at the spec.
