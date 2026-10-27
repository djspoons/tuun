---
type: grilling
status: closed
assignee: djspoons
blocked_by: []
---

# What do Cmd+Enter and Shift+Cmd+Enter do once Loop is per program?

## Question

The map's charting decisions only covered bottom pads, Play and evaluate-and-play. But the keyboard also has launches that pick their own period, found while writing the [spec](../spec.md):

- In Select mode, Cmd+Return plays at the next measure and loops every 1 measure, and Shift+Cmd+Return loops every 2 measures. This is `Action::PlayProgram` with `repeat_after_measures: Some(1 | 2)` in `sdl2_input.rs`.
- In Edit mode, Cmd+Return and Shift+Cmd+Return evaluate, then do the same. Plain Return evaluates and plays once. This is `Action::EvaluateAndLeaveEditMode` with `repeat_after_measures`.

The README documents all of these under "Keyboard Navigation".

Once `loop_after_beats` lives on the program and there is no global setting, what should each of these keys do?

- Does Cmd+Return honour the annotation, as the Play button and bottom pads do?
- Is Shift+Cmd+Return dropped, or given another meaning?
- Does plain Return in Edit mode keep playing once, or honour the annotation too? The charting note says evaluate-and-play honours Loop, but plain Return plays once today.
- If any key keeps a per-launch period that overrides the annotation, is that still "a program's own setting" as the glossary's **Loop** defines it?

## Resolution

Every launch either starts at the next measure and follows the program's Loop, or starts now and plays once. No key or pad picks its own period anymore.

| Launch | Starts | Loops? |
| --- | --- | --- |
| Bottom pad, Launchkey Play, Edit-mode Return, finishing a recording with Play | next measure (a recording: the take's end boundary) | as the program's `loop_after_beats` says, or once without it |
| Top pad (Toggle or Trigger), Cmd+Return in Select or Edit mode | now | never |

- **Cmd+Return** is the keyboard's immediate launch, in both Select and Edit modes. Edit mode evaluates first.
  - It plays the whole program once, starting now. For a sequence, the pattern's first beat lands at the press, so it's offset from the barline, as a Toggle-mode top pad already does.
  - It never stops what is already sounding, whatever the launch mode, so pressing it twice layers two plays.
- **Return in Edit mode** evaluates, then makes a pending launch that follows the annotation. In Select mode, Return still enters Edit mode, so Select mode has no single key for a pending launch. Use the Launchkey's Play button or bottom pads, or press Return twice.
- **Shift+Cmd+Return** is dropped in both modes. Anyone who wants a loop every other measure writes `loop_after_beats=8` in 4/4.
- **Finishing a recording with Play** queues the take at its end boundary, following the slot's own `loop_after_beats`. Recording replaces only the program's text, so the slot keeps its annotations.
- **A Trigger-mode top pad on a sequence** plays the whole pattern from now and adds a pass on each press. This replaces the one-hit special case. `Action::StartProgramVoice`, `Effect::PlayProgramVoice` and `Player::play_program_voice` were its only path, so they are deleted. Every immediate launch then goes through `play_program(…, Start::Now, None)`. To drum a pattern's hit on its own, give the hit a program slot of its own.
- **Messages.** "Playing voice of kick" came only from `play_program_voice`, so immediate launches show "Playing waveform kick" or "Playing sequence kick". This replaces the immediate-launch line in [How does tuun's screen show a program's Loop?](06-loop-on-screen.md). Pending launches keep that ticket's wording.
- **Corrections.** The charting note's "evaluate-and-play" means Edit-mode Return. The glossary's **Trigger** is redefined as "always starts its program again, a whole pattern for a sequence".

Considered and dropped:
- making Cmd+Return a pending launch in Select mode only, which would give one key different meanings in the two modes
- moving the pending launch onto Shift+Cmd+Return
- keeping a period that overrides the annotation on any key
- keeping the one-hit Trigger
