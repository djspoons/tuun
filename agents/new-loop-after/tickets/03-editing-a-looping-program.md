---
type: grilling
status: closed
assignee: djspoons
blocked_by: [02]
---

# When does a changed Loop take effect on a program that is already looping?

## Question

If a program's Loop is edited (from the keyboard, from the Launchkey, or by editing its text) while it is looping, does the new value apply to the voices already scheduled, at the next play, at the next measure, or only at the next launch? What happens when the annotation is removed while the program is looping: does it stop after the current play? Is the answer the same for sequencer programs? Since loops drift off the barline (see [How do loops that aren't a whole number of measures behave?](02-loops-off-the-measure.md)), "the next play" and "the next measure" are usually different moments.

## Resolution

A Loop change takes effect at the program's next launch. Nothing already sounding or queued changes. This holds however the Loop was changed: from the keyboard (`Cmd-N l`, see [Which prefixed commands edit Loop from the keyboard?](04-keyboard-loop-commands.md)), from the Launchkey, or on disk. A reload stops every voice and clears every queued play anyway.

- **Changing N.** `kick` loops every 4 beats and `Cmd-N l 3` is pressed. It keeps looping every 4 beats until it's stopped and launched again.
- **Removing Loop.** `Cmd-N l 0` on a program that's looping only rewrites the source. The program keeps looping until the bottom pad or Stop ends it, and its next launch plays once.
- **Adding Loop.** Adding Loop to a program that's playing once only writes the source, and the next launch loops.
- **Sequences.** A step turned on while a sequence loops takes the period that's already running (the gap between the current pass and the next queued one), not the edited annotation. This is what `Player::play_step` does today. A step edit before the first pass uses the annotation, as decided in [How do loops that aren't a whole number of measures behave?](02-loops-off-the-measure.md).
- **No stacked loops.** Entering Edit mode removes the program's queued plays, future loop plays included, and the one that's sounding finishes (`Action::EnterEditMode` calls `remove_pending_effects`). Play and the bottom pad refuse to queue when something is already queued. So a program never has two loops queued, and evaluate-and-play needs no extra guard. All of this is today's behaviour and stays as it is.
- **Message.** When a Loop change doesn't affect a program that's playing, the message ends "from its next launch", for example "kick loops every 3 beats from its next launch". When the program isn't playing, the message uses ticket 04's plain wording. The rest of the wording belongs to how the screen shows Loop (still unspecified on the map).
