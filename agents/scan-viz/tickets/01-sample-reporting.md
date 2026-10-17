# How the audio thread reports samples

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

How should the tracker report output samples to the UI so that both Snapshot and Scan can be drawn, replacing the `Command::SendCurrentBuffer` / `next_buffer_refresh` polling in `src/main.rs`?

Options to weigh:

- A summary (for example, min/max per slice) in every `Status`, which the tracker already sends after each callback.
- A request from the UI that sets the window and resolution.
- A mark-style request carrying main-thread timing such as measure starts. `MarkId` has commented-out `VisualizeTiming` and `ShowSample` variants. Marks today are tied to one waveform, while the scopes show the overall output.

Constraints: the scopes show the overall output; nothing new allocates on the audio thread; the change is small and surgical. Informed by [Lock-free handoff from audio thread to UI in Rust](02-lock-free-handoff.md) when it is resolved.

## Resolution

The tracker pushes every callback's raw output samples to the UI through two lock-free `rtrb` rings. This replaces `Command::SendCurrentBuffer`, `Status.buffer` and `next_buffer_refresh`.

- **Raw samples cross the boundary.** The audio thread only copies samples; the UI computes Scan's per-slice min/max. The slice width stays a UI choice.
- **Push, every callback,** whether or not a scope is drawn.
- **Two rings,** both allocated once at startup:
  - a sample ring of `f32`, holding one second of audio (`sample_rate` samples);
  - a header ring of plain-value chunk headers: `buffer_start: Instant`, samples written, and samples dropped since the previous header. It is sized so it practically never fills before the sample ring (for example, 1024 headers).
- **Protocol:**
  - The audio thread checks that the header ring has a free slot and the sample ring has room for the whole buffer. If so, it pushes the samples, then the header. Otherwise it pushes nothing and adds the buffer's length to a running dropped count, which the next header carries.
  - The UI pops a header, then reads exactly the number of samples it names.
  - Seeing a header guarantees its samples are visible. Verified by reading `rtrb` 0.4.0's source (not by running it): its pushes end with a `Release` store of the tail, and its pops and chunk reads refresh the tail with an `Acquire` load when their cached copy doesn't show enough items ([research](../research/lock-free-handoff.md)).
- **Draining:** the UI drains both rings on every main-loop pass, next to the status drain. Like statuses, samples don't wake the loop, which wakes at least once a frame for its next render. Removing `next_buffer_refresh` also removes it from the loop's wait deadline (`next_render.min(next_buffer_refresh)` in `src/main.rs`).
- **When the UI falls behind,** the newest samples are dropped and the gap is recorded, so Scan can draw the gap rather than squash time. A source reload still blocks the main thread for about 2.6 s (measured on the main-thread work map, `agents/main-thread/` on `claude/main-thread`), longer than the one-second sample ring, so a reload shows as a gap. That is intended; the ring stays one second.
- **Timing:** each chunk's `buffer_start` places its samples on the clock, so the UI can align them with measure starts. No mark-based timing; `MarkId::VisualizeTiming` stays commented out.
- **Allocation:** only the new path must be allocation-free. The rest of `Status` (including the marks `Vec` and the `mpsc` send) is unchanged.
- **Snapshot** shows the latest N samples from the stream, decoupled from `--buffer-size`. N is decided in [Scan look](04-scan-look.md).
- **New dependency:** `rtrb`.

