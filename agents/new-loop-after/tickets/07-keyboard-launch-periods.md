---
type: grilling
status: open
assignee:
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
