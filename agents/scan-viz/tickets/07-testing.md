# Testing the sample path and scope views

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

How should the new sample path and scope views be tested? In particular:

- How are the tracker's all-or-nothing push and its dropped-sample count tested without an audio device, given that `src/lib/tracker.rs` already has unit tests that drive `Tracker::callback` directly?
- How is the UI side tested: popping a header and then its samples, placing chunks on the clock, and keeping the latest 2048 samples for Snapshot?
- Does scope rendering (Snapshot, the Scan sweep, clipping ticks, gaps) get automated tests at all? `src/lib/renderer.rs` has none today, or should it be split so the per-slice min/max and layout arithmetic can be tested apart from SDL drawing?
- Which on-screen metrics serve as acceptance checks on the dev's machine? The tracker's `allocations` metric counts buffers allocated per callback, so the new path should leave it unchanged. `renderer_load` (time from a frame's due time to present, as a fraction of `FRAME_BUDGET`) shows whether Scan, which draws one slice per pixel column every frame, fits in a 60 Hz frame. Ordinary renders already take 5–6 ms (main-thread work map).

## Resolution

- **Tracker push: unit tests in `src/lib/tracker.rs`.** `Tracker::new` takes the two ring producers, so each test builds its own rings and can make them tiny. Tests call `callback` directly, as the existing ones do, and check three cases:
  1. the samples, then their header, arrive in order, with the right `buffer_start` and sample count;
  2. a full sample ring means nothing is pushed, and the next header that gets in reports the dropped count;
  3. a full header ring, with room in the sample ring, also means nothing is pushed.
- **Allocation-free push: checked by inspection.** No counting-allocator test.
- **UI side: a new module with no drawing, for example `src/lib/scope.rs`, unit-tested with rings and headers built in the tests.** It holds everything that isn't SDL drawing:
  - draining a header and then exactly its samples;
  - keeping the latest 2048 samples for Snapshot;
  - turning chunks into per-column min/max for Scan, placed by time within the two-measure window;
  - marking gaps from dropped counts;
  - choosing a column's clipping colour (none, yellow above 0.95, red above 1.0).
  `src/lib/renderer.rs` only draws what it is given.
- **SDL drawing: no automated tests.** It is checked by eye.
- **Acceptance on the dev's machine:**
  - the tracker's `allocations` metric is unchanged from before;
  - render time and `renderer_load` stay within the 60 Hz frame budget with Scan showing;
  - by eye: an envelope sweeps across two measures; clipping ticks show in yellow and red; Ctrl+S switches views and shows the status message; `--scope-view scan` starts in Scan; `--snapshot-refresh-ms` changes how often Snapshot and the spectrum redraw.
- **`--show-loop-timing` is temporary.** Its level of detail doesn't need to outlive the main-thread work. The acceptance check relies on the metrics already on `main` (`allocations`, `renderer_load`). If the loop timing is needed, it is brought in as one self-contained commit (from `claude/main-thread`) and reverted afterwards; nothing in this work depends on it.
- **Gaps are checked by unit tests only.** A source reload no longer stalls the main thread (it runs on the processing thread on `main`), so there is no easy way to make a gap by hand; the `scope.rs` tests cover turning dropped counts into gap columns.
