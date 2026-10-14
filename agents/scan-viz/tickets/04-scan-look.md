# Scan look

Labels: wayfinder:prototype
Status: closed
Assignee: djspoons
Blocked by: [How the audio thread reports samples](01-sample-reporting.md), [Scan's time axis: measures or seconds](03-scan-time-axis.md)

## Question

How should Scan look on screen: window length, scrolling or sweeping, linear or dB amplitude, how clipping is coloured, which gridlines are drawn, and how many recent samples (N) Snapshot shows? Resolve by building a rough working view to react to.

Prototype: [prototypes/scan-look.html](../prototypes/scan-look.html) (open in a browser; ←/→ switch variants, Tab switches scope view).

## Resolution

Decided by reacting to the [prototype](../prototypes/scan-look.html):

- **Layout: sweep.** A write head moves left to right across the window and wraps; the previous pass stays dimmed ahead of it.
- **Window: two measures.** No minimum length in seconds.
- **Amplitude: linear.**
- **Clipping:** ticks along the top edge, yellow where a slice's peak is above 0.95 and at most 1.0, red above 1.0. These are Snapshot's existing thresholds and colours.
- **Gridlines:** measure lines only.
- **Snapshot:** the latest N = 2048 samples.
- **Dropped samples:** drawn as a hatched gap, as in the prototype (not separately discussed).

