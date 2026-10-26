---
type: grilling
status: closed
assignee: djspoons
blocked_by: []
---

# Which prefixed commands edit Loop from the keyboard?

## Question

Following the `Cmd-N` annotation prefix (`PrefixKey::Annotation`, prompt "Annotation? (k)eys (u)p (d)own"), which keys raise, lower, and clear the active program's `loop_after_beats`? Is that one key per action, or a sub-prompt that takes a number? What step size is used (one beat, or one measure's worth)? What happens at the bounds (below 1 clears it?), what does the prompt say, and what message confirms the change? Which modes accept it, and how does it interact with a `#{keys}` program (where the annotation is an error)?

## Resolution

- **Two prefixes in a row.** `Cmd-N` (`PrefixKey::Annotation`) gains `l`, and its prompt becomes "Annotation? (k)eys (u)p (d)own (l)oop". `l` starts a second prefix, Loop, which sits beside Annotation and prompts "Loop kick? (1-9) beats (0) once".
- **The next key completes it.** A digit `1`–`9` sets `#{loop_after_beats=N}` and shows "kick loops every N beats". `0` removes the annotation and shows "kick plays once". Any other key cancels, as with any prefix. A modifier press or a held key's auto-repeat does neither. Only one digit is entered: a typed-number field was considered and dropped.
- **Modes.** The same as `Cmd-N`: Select and Keys only. In Keys mode the key that completes the Loop prefix plays no note.
- **Guards, checked when `l` is pressed.** The Loop prefix doesn't start on an empty slot ("B:3 is empty"), on stale source (the stale-source message), or on a `#{keys}` program ("kick plays keys, so it can't loop"). The digit is the very next key press, so nothing is checked again.
- **Keys and Loop.** `Cmd-N k` on a program that has a Loop is refused with "Remove kick's Loop first", so no setting is lost that wasn't asked to change.
- **Writing back.** Each change is written to the source the way `k` writes `#{keys}`.
- **Range.** The keyboard reaches 1–9 beats. The source accepts any positive whole number of beats, with no upper bound.
- **Names.** Messages name the program through `display_name`, so an unnamed program reads "B:3 loops every 8 beats".
- **Not decided here.** What a change does to a program that is already playing is left to [When does a changed Loop take effect on a program that is already looping?](03-editing-a-looping-program.md).
