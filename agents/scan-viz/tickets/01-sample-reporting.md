# How the audio thread reports samples

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: none

## Question

How should the tracker report output samples to the UI so that both Snapshot and Scan can be drawn, replacing the `Command::SendCurrentBuffer` / `next_buffer_refresh` polling in `src/main.rs`?

Options to weigh:

- A summary (for example, min/max per slice) in every `Status`, which the tracker already sends after each callback.
- A request from the UI that sets the window and resolution.
- A mark-style request carrying main-thread timing such as measure starts. `MarkId` has commented-out `VisualizeTiming` and `ShowSample` variants. Marks today are tied to one waveform, while the scopes show the overall output.

Constraints: the scopes show the overall output; nothing new allocates on the audio thread; the change is small and surgical. Informed by [Lock-free handoff from audio thread to UI in Rust](02-lock-free-handoff.md) when it is resolved.
