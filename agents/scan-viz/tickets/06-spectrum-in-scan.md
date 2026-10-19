# The spectrum in Scan

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

What happens to the spectrum (the FFT drawn below the scope) while Scan is shown: kept as it is (computed from the latest Snapshot samples), hidden, or replaced so Scan gets the full screen height? Scan sweeps across two measures in the top of the screen, with ticks for clipping along its top edge ([Scan look](04-scan-look.md)).

## Resolution

- **Kept in Scan.** The spectrum stays below the scope in both scope views, unchanged, computed from the latest Snapshot samples (N = 2048). Scan keeps the top 3/5 of the screen.
- **Redraw rate:** Snapshot and the spectrum redraw on a UI timer, not every frame, so an untriggered Snapshot doesn't jitter. A CLI flag sets the interval, for example `--snapshot-refresh-ms`, defaulting to 100 ms. (It was first set to 200 ms, the old rate, and lowered to 100 ms after trying it.) The timer asks nothing of the audio thread. The main loop already renders at most once a frame (`renderer::FRAME_BUDGET`, 60 Hz), so the timer only decides, when a frame renders, whether to refresh Snapshot's samples and recompute the spectrum. It adds no wake-up deadline of its own and fires up to one frame late. Scan is unaffected and moves on every render.

