---
type: grilling
status: closed
assignee: djspoons
blocked_by: []
---

# How do repeats that aren't a whole number of measures behave?

## Question

A launch starts on the next measure boundary, but `repeat_after_beats=3` in 4/4 makes later plays land off the barline. Is that drift allowed, and does anything snap back to the measure? Leaning: allow drift. How does this interact with sequencer programs (`[steps] | on_beats(w)`), which schedule one independently repeating voice per step at measure-relative beats (`player.rs`)? What about a Repeat shorter than the waveform's length, or a change of `beats_per_measure` or tempo while it repeats?

## Resolution

A program with `repeat_after_beats=N` plays again every N beats, counted from the measure boundary where it was launched, and the same rule covers waveform and sequence programs. Nothing special-cases a Repeat that isn't a whole number of measures, or one that is shorter than the program:

- **Drift is allowed.** Nothing snaps a repeating program back to the barline, rounds N to whole measures, or restricts which N are allowed. For example, `repeat_after_beats=3` in 4/4 lands on beats 1, 4, 3, 2, and returns to the barline every 12 beats.
- **Plays overlap.** When a play is still sounding as the next one starts, both sound. A new play never cuts off the previous one, and nothing warns about it. This is the tracker's behaviour today.
- **Sequences play as written.** A sequence whose steps extend past beat N plays all of its steps every pass, so the passes overlap. For example, `[1, 2, 3, 4] | on_beats(kick)` with `repeat_after_beats=2` doubles up: beats 3 and 4 of one pass land on beats 1 and 2 of the next. Steps aren't dropped, and the program isn't rejected.
- **Step edits before the first pass.** A step edited before a sequence's first pass has started takes its period from that program's own `repeat_after_beats`. Without the annotation it plays once. Today this period comes from the app-wide setting, in `Player::play_step`'s `default_repeat_measures`.
- **Tempo and measure length.** Tempo and beats per measure are fixed for a session, as command-line options with no runtime setter, so the spec says nothing about changing them while a program repeats. Live tempo is out of scope. A Repeat counts beats, so a future live tempo would naturally carry it along.
