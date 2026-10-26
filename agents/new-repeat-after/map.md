---
label: wayfinder:map
---

# Per-program Repeat, written as an annotation

## Destination

A spec, ready to implement in one go, for replacing the app-wide Repeat setting with a per-program `repeat_after_beats=N` annotation. It covers syntax, semantics, which launches honour it, editing from the keyboard and from the Launchkey, and the doc and glossary changes.

## Notes

- Domain: the tuun live instrument. Use the vocabulary in [`agents/CONTEXT.md`](../CONTEXT.md); **Repeat** there is already redefined for this effort. For HITL tickets, call the Skill tool for "grilling" and "domain-modeling".
- Plan, don't do: tickets resolve decisions. Code changes belong to the implementation that follows the map.
- Settled while charting (no ticket):
  - The app-wide setting (`AppState.repeat_after_measures`, `Action::CycleRepeatAfterMeasures`, the Function-pad cycle and its colours) is removed entirely. There is no global fallback. A program without the annotation plays once.
  - Spelling is `#{repeat_after_beats=N}`, where N is a positive whole number of beats.
  - `repeat_after_beats` on a `#{keys}` program is an error.
  - Only launches that start at the next measure repeat. This keeps today's split: pending launches (bottom pads, Play, evaluate-and-play) honour Repeat, and immediate launches (top pads, `ToggleProgramPlayback`) play once.
  - Repeat can be edited live, both from the keyboard and from the Launchkey, and is written back to the annotation. The `Cmd-N` annotation prefix command (`PrefixKey::Annotation`) is the guide.
  - Docs are part of the spec (README Launchkey section, language spec § Annotations, glossary). There is no data migration, because the global setting was never persisted.

## Decisions so far

<!-- one line per closed ticket: - [<title>](tickets/NN-slug.md): <gist> -->

## Not yet specified

- What the Launchkey's Function pad does once it no longer cycles Repeat: freed, repurposed, or folded into the Repeat-button design.
- How the screen shows a program's Repeat (program list, status line, launch message wording such as today's "Looping waveform B:3 every 4 beats").
- Whether the annotation needs a practical upper bound, beyond what the keyboard and Launchkey editors can reach.

## Out of scope

- Fractional beats (e.g. `repeat_after_beats=1.5`): whole beats only.
- Keeping an app-wide Repeat as a default for programs without the annotation.
- Making immediate (top-pad) launches repeat.
