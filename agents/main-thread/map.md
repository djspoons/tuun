# Main-thread work and input latency

Labels: wayfinder:map

## Destination

A spec, in this directory, that ranks the work on the native UI's main thread by its effect on input latency, with a decision for each item: keep it there, cache it, or move it to another thread.

Draft: [spec.md](spec.md).

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
- [Font loading per frame](tickets/05-font-loading.md): fonts are loaded once and held by `Renderer<'ttf>`. That is estimated to save 1–4 ms per frame, most of it, by hypothesis, from the glyph cache that each fresh font throws away.
- [The 10 ms status wait](tickets/04-status-wait.md): removed. The loop blocks in SDL's `wait_event_timeout`; SDL input, Launchkey events and processing results wake it, statuses don't. It renders at most 60 Hz and sleeps until the next render, tick or buffer refresh, so an input waits only behind a render in progress (target: about 6 ms max keyboard latency).
- [Measure render cost](tickets/08-measure-render.md): `present()` doesn't wait (Metal, vsync off); text textures are about two thirds of a 5–6 ms render at about 0.14–0.19 ms each; Edit mode on a long program needs about 112 and takes about 22.8 ms, over a 60 Hz frame.
- [Caching text textures](tickets/10-text-textures.md): no cache; the active Edit-mode line is drawn in runs of one colour instead of one texture per character, with the cursor placed by `font.size_of`; acceptance is an Edit-mode render under 16.7 ms.
- [Measure the Edit-mode helpers](tickets/09-measure-edit-helpers.md): the per-keystroke parse and ShowType stay on the main thread (at most 0.5 and 2.5 ms); completion and parameter hints take 69–317 ms in release and 193–920 ms in debug on a large file.
- [Where completion and parameter hints run](tickets/11-completion.md): entering Edit mode prefetches the program's context on the processing thread; Cmd-/ looks names up in it, or shows "Preparing completions…" until it arrives. Only the context of the program being edited is kept; reload refetches it, slider moves don't.

## Not yet specified

- A text texture cache kept across frames, if later measurements show ordinary frames over budget.
- Whether note-on stays within budget as keys instruments get heavier, and for chords in debug builds (a four-note flute chord is about 8–10 ms).

## Out of scope

- The audio thread, the precompute thread, and the web/wasm UI.
- Making the moves themselves: this map ends at the spec.
- Caching evaluated modules and bindings on the processing thread: it shortens the "Processing…" wait, not input latency ([Caching evaluated modules and bindings](tickets/07-evaluation-caching.md)).
- The hard-coded macOS font path in `src/lib/renderer.rs`: portability, not latency ([Font loading per frame](tickets/05-font-loading.md)).
