# Testing the sample path and scope views

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: none

## Question

How should the new sample path and scope views be tested? In particular:

- How are the tracker's all-or-nothing push and its dropped-sample count tested without an audio device, given that `src/lib/tracker.rs` already has unit tests that drive `Tracker::callback` directly?
- How is the UI side tested: popping a header and then its samples, placing chunks on the clock, and keeping the latest 2048 samples for Snapshot?
- Does scope rendering (Snapshot, the Scan sweep, clipping ticks, gaps) get automated tests at all? `src/lib/renderer.rs` has none today, or should it be split so the per-slice min/max and layout arithmetic can be tested apart from SDL drawing?
