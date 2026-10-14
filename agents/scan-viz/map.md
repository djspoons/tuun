# Scope views: Snapshot and Scan

Labels: wayfinder:map

## Destination

A spec, in this directory, for adding the Scan scope view alongside Snapshot in the native UI, ready to hand to an implementation session.

## Notes

- Domain: the native SDL2 UI (`src/lib/renderer.rs`, `src/main.rs`) and the tracker's reporting of samples to it (`src/lib/tracker.rs`).
- Follow `CLAUDE.md`. Use the terms **Scope view**, **Snapshot** and **Scan** from `agents/CONTEXT.md`.
- Keep work off the audio thread unless it's allocation-free.
- Prefer small, surgical changes to how the audio thread reports samples.
- Tickets and their workflow: see [README.md](README.md).

## Decisions so far

- [Lock-free handoff from audio thread to UI in Rust](tickets/02-lock-free-handoff.md): mpsc sometimes allocates and can lock. `rtrb`/`ringbuf` are lock-free with a fixed capacity and drop the newest data when full; `triple_buffer` keeps only the latest; `ArrayQueue::force_push` drops the oldest.

## Not yet specified

- Where the spectrum belongs in Scan: kept, hidden, or replaced. Waits on how Scan looks.
- How to test scope rendering and the sample-reporting protocol.

## Out of scope

- Per-program scopes: both views always show the overall output mix.
- The web/wasm UI.
- Snapshot triggering (for example, aligning to zero crossings).
- Freezing or pausing the scope.
