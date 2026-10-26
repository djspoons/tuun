---
type: grilling
status: open
assignee:
blocked_by: []
---

# How do repeats that aren't a whole number of measures behave?

## Question

A launch starts on the next measure boundary, but `repeat_after_beats=3` in 4/4 makes later plays land off the barline. Is that drift allowed, and does anything snap back to the measure? Leaning: allow drift. How does this interact with sequencer programs (`[steps] | on_beats(w)`), which schedule one independently repeating voice per step at measure-relative beats (`player.rs`)? What about a Repeat shorter than the waveform's length, or a change of `beats_per_measure` or tempo while it repeats?
