# Lock-free handoff from audio thread to UI in Rust

Labels: wayfinder:research
Status: open
Assignee:
Blocked by: none

## Question

What are the established ways to hand audio-rate data (whole buffers or per-slice summaries) from a real-time audio callback to a UI thread in Rust, and what does each cost compared with the `std::sync::mpsc` channel the tracker uses today? In particular: does sending allocate, is it lock-free or wait-free, and what happens when the UI falls behind? Candidates include the `rtrb` and `ringbuf` crates and triple buffers.
