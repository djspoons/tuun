# Lock-free handoff from audio thread to UI in Rust

Labels: wayfinder:research
Status: closed
Assignee: djspoons
Blocked by: none

## Question

What are the established ways to hand audio-rate data (whole buffers or per-slice summaries) from a real-time audio callback to a UI thread in Rust, and what does each cost compared with the `std::sync::mpsc` channel the tracker uses today? In particular: does sending allocate, is it lock-free or wait-free, and what happens when the UI falls behind? Candidates include the `rtrb` and `ringbuf` crates and triple buffers.

Findings: [research/lock-free-handoff.md](../research/lock-free-handoff.md)

## Resolution

- `std::sync::mpsc::channel()` is unbounded and not real-time safe. `send` allocates a new block once every 31 messages. It can also take a mutex and make a wake-up syscall when the UI is waiting in `recv_timeout`.
- `rtrb` and `ringbuf` are lock-free single-producer, single-consumer ring buffers that allocate only once, when created. When either is full, the push fails and the newest data is dropped. `rtrb` can push whole slices of `Copy` values, such as `f32` samples, in one call.
- `triple_buffer` publishes only the latest value with one atomic swap, so anything the UI doesn't read in time is lost.
- `crossbeam_queue::ArrayQueue::force_push` drops the oldest item instead.
- Whichever queue is used, only payloads of plain values with no heap memory (raw samples, fixed arrays, min/max pairs) keep the audio thread allocation-free. Today's `Status` allocates whenever it has marks, and the `buffer` copy allocates every 200 ms.
- Found by reading the code, not by running it: the drain loop in `main.rs` keeps only the last `Status`, so a `buffer` can be overwritten before it is drawn.

Details and sources are in [research/lock-free-handoff.md](../research/lock-free-handoff.md).
