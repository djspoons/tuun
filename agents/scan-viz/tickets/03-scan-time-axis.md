# Scan's time axis: measures or seconds

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: [How the audio thread reports samples](01-sample-reporting.md)

## Question

Should Scan's time axis follow musical time (a window of N measures that restarts on a measure boundary, with beat lines) or wall-clock time (a fixed number of seconds, independent of tempo)?

## Resolution

**Musical time.** Scan's window is a whole number of measures and lines up with measure boundaries, so a measure always takes the same width on screen.

- Tempo and beats per measure are fixed for the whole run (`--tempo`, `--beats_per_measure`), so the window's length in seconds is fixed for a run too.
- The UI already knows where beats fall (`current_beat_info` in `src/lib/renderer.rs`, from the Beats marks), and each sample chunk carries its `buffer_start`. Aligning to measures needs no new reporting.
- How many measures, and whether a minimum length in seconds guards very fast tempos, is left to [Scan look](04-scan-look.md).

