---
type: grilling
status: closed
assignee: djspoons
blocked_by: []
---

# How do loops that aren't a whole number of measures behave?

## Question

A launch starts on the next measure boundary, but `loop_after_beats=3` in 4/4 makes later plays land off the barline. Is that drift allowed, and does anything snap back to the measure? Leaning: allow drift. How does this interact with sequencer programs (`[steps] | on_beats(w)`), which schedule one independently looping voice per step at measure-relative beats (`player.rs`)? What about a Loop shorter than the waveform's length, or a change of `beats_per_measure` or tempo while it loops?

## Resolution

A program with `loop_after_beats=N` plays again every N beats, counted from the measure boundary where it was launched, and the same rule covers waveform and sequence programs. Nothing special-cases a Loop that isn't a whole number of measures, or one that is shorter than the program:

- **Drift is allowed.** Nothing snaps a looping program back to the barline, rounds N to whole measures, or restricts which N are allowed. For example, `loop_after_beats=3` in 4/4 lands on beats 1, 4, 3, 2, and returns to the barline every 12 beats.
- **Plays overlap.** When a play is still sounding as the next one starts, both sound. A new play never cuts off the previous one, and nothing warns about it. This is the tracker's behaviour today.
- **Sequences play as written.** A sequence whose steps extend past beat N plays all of its steps every pass, so the passes overlap. For example, `[1, 2, 3, 4] | on_beats(kick)` with `loop_after_beats=2` doubles up: beats 3 and 4 of one pass land on beats 1 and 2 of the next. Steps aren't dropped, and the program isn't rejected.
- **Step edits before the first pass.** A step edited before a sequence's first pass has started takes its period from that program's own `loop_after_beats`. Without the annotation it plays once. Today this period comes from the app-wide setting, in `Player::play_step`'s `default_repeat_measures`.
- **Tempo and measure length.** Tempo and beats per measure are fixed for a session, as command-line options with no runtime setter, so the spec says nothing about changing them while a program loops. Live tempo is out of scope. A Loop counts beats, so a future live tempo would naturally carry it along.
