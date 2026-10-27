---
label: wayfinder:map
---

# Per-program Loop, written as an annotation

## Destination

A spec, ready to implement in one go, for replacing the app-wide Loop setting with a per-program `loop_after_beats=N` annotation. It covers syntax, semantics, which launches honour it, editing from the keyboard and from the Launchkey, and the doc and glossary changes.

## Notes

- Domain: the tuun live instrument. Use the vocabulary in [`agents/CONTEXT.md`](../CONTEXT.md); **Loop** there is already redefined for this effort. For HITL tickets, call the Skill tool for "grilling" and "domain-modeling".
- Plan, don't do: tickets resolve decisions. Code changes belong to the implementation that follows the map.
- Settled while charting (no ticket):
  - The app-wide setting (`AppState.repeat_after_measures`, `Action::CycleRepeatAfterMeasures`, the Function-pad cycle and its colours) is removed entirely. There is no global fallback. A program without the annotation plays once.
  - Spelling is `#{loop_after_beats=N}`, where N is a positive whole number of beats.
  - `loop_after_beats` on a `#{keys}` program is an error.
  - Only launches that start at the next measure honour Loop. This keeps today's split: pending launches (bottom pads, Play, Edit-mode Return) honour Loop, and immediate launches (top pads, Cmd+Return) play once. See [ticket 07](tickets/07-keyboard-launch-periods.md) for the full list.
  - Loop can be edited live, both from the keyboard and from the Launchkey, and is written back to the annotation. The `Cmd-N` annotation prefix command (`PrefixKey::Annotation`) is the guide.
  - Docs are part of the spec (README Launchkey section, language spec § Annotations, glossary). There is no data migration, because the global setting was never persisted.

## Decisions so far

<!-- one line per closed ticket: - [<title>](tickets/NN-slug.md): <gist> -->

- [Which Launchkey control should carry the Loop prefix, and what can we do with it?](tickets/01-launchkey-loop-button.md): The MK4 37 has no "Repeat" button. The candidates are Loop (CC 118, single-colour LED) and Function (CC 105, RGB). 16 RGB pads are available, and Shift+pad is reserved by the firmware.
- [How do loops that aren't a whole number of measures behave?](tickets/02-loops-off-the-measure.md): A program plays again every N beats from its launch boundary. It drifts freely off the barline, and overlapping plays (including sequence passes) all sound. A pre-first-pass step edit takes the program's own Loop.
- [Which prefixed commands edit Loop from the keyboard?](tickets/04-keyboard-loop-commands.md): `Cmd-N l` starts a Loop prefix. Then `1`–`9` sets that many beats, `0` makes the program play once, and any other key cancels. It works in Select and Keys modes and is refused on Keys programs, and `Cmd-N k` is refused on a program that loops. The source has no upper bound.
- [When does a changed Loop take effect on a program that is already looping?](tickets/03-editing-a-looping-program.md): A change takes effect at the next launch, whether it adds, changes or removes the Loop. Whatever is sounding or queued keeps its period, and steps added to a looping sequence take the running period. While the program plays, the message adds "from its next launch".
- [How do the Launchkey's Loop prefix and pads display and set Loop?](tickets/05-launchkey-loop-prefix.md): Holding the Loop button is a temporary DAW pad sub-mode. The top row is 1–8 beats and the bottom row is 1–8 measures. The current value is white and the rest are lit dimly in the program's colour. The last pad pressed wins, pressing the white pad again means once, and the value is written when Loop is released. The Function button is freed and unlit.
- [How does tuun's screen show a program's Loop?](tickets/06-loop-on-screen.md): Programs other than the active one show `↻` (Loop) or `⌨` (keys) in the gutter, in the program's colour, following the annotation. The active program keeps the triangle. Selecting a program shows e.g. "B:3 kick, loops every 4 beats". Launch messages keep today's wording with N from the annotation, and the status line mirrors a held Launchkey Loop edit.
- [What do Cmd+Enter and Shift+Cmd+Enter do once Loop is per program?](tickets/07-keyboard-launch-periods.md): Pending launches follow Loop: bottom pads, Play, Edit-mode Return, and finishing a recording with Play. Immediate launches play the whole program once and never stop what's sounding: top pads and Cmd+Return. That includes a Trigger-mode pad on a sequence, which now plays the pattern. Shift+Cmd+Return is dropped.

## Spec

[spec.md](spec.md): ready to implement.

## Not yet specified

- (none: every remaining question is a ticket)

## Out of scope

- Fractional beats (e.g. `loop_after_beats=1.5`): whole beats only.
- Keeping an app-wide Loop as a default for programs without the annotation.
- Making immediate (top-pad) launches loop.
- Changing tempo or beats per measure while programs loop. Both are fixed per session, so this belongs to whatever effort makes tempo live ([ticket](tickets/02-loops-off-the-measure.md)).
