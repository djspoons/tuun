# Scope views: Snapshot and Scan

Labels: wayfinder:map

## Destination

A spec, in this directory, for adding the Scan scope view alongside Snapshot in the native UI, ready to hand to an implementation session.

## Notes

- Domain: the native SDL2 UI (`src/lib/renderer.rs`, `src/main.rs`) and the tracker's reporting of samples to it (`src/lib/tracker.rs`).
- Follow `CLAUDE.md`. Use the terms **Scope view**, **Snapshot** and **Scan** from `agents/CONTEXT.md`.
- Keep work off the audio thread unless it's allocation-free.
- Prefer small, surgical changes to how the audio thread reports samples.
- Bind app-state and other less common changes to Ctrl key combinations.
- Tickets and their workflow: see [README.md](README.md).

## Decisions so far

- [Lock-free handoff from audio thread to UI in Rust](tickets/02-lock-free-handoff.md): mpsc sometimes allocates and can lock. `rtrb`/`ringbuf` are lock-free with a fixed capacity and drop the newest data when full; `triple_buffer` keeps only the latest; `ArrayQueue::force_push` drops the oldest.
- [How the audio thread reports samples](tickets/01-sample-reporting.md): raw samples pushed every callback through an `rtrb` sample ring plus a header ring (timestamp, written, dropped), all-or-nothing per callback; newest dropped when full; replaces `SendCurrentBuffer`; Snapshot becomes the latest N samples.
- [Scan's time axis: measures or seconds](tickets/03-scan-time-axis.md): musical time; the window is a whole number of measures aligned to measure boundaries.
- [Scan look](tickets/04-scan-look.md): a sweep over two measures, linear amplitude, clipping ticks on the top edge in Snapshot's yellow and red, measure lines only; Snapshot shows the latest 2048 samples.
- [Switching scope views](tickets/05-switching-scope-views.md): Ctrl+S toggles in every editor mode where possible; Snapshot by default; `--scope-view` CLI flag sets the startup view; a status message on each switch; no Launchkey button.
- [The spectrum in Scan](tickets/06-spectrum-in-scan.md): kept below the scope in both views; Snapshot and the spectrum redraw on a UI timer set by a CLI flag (default 200 ms).

## Not yet specified


## Out of scope

- Per-program scopes: both views always show the overall output mix.
- The web/wasm UI.
- Snapshot triggering (for example, aligning to zero crossings).
- Freezing or pausing the scope.
- Moving existing app-state key bindings to Ctrl combinations: a standing preference, not part of this effort.
