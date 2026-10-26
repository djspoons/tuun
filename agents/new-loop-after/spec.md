# Spec: per-program Loop, written as an annotation

This spec is the destination of the [Per-program Loop map](map.md). Each section says which ticket decided it. The ticket holds the reasoning and the options that were rejected. Words follow the glossary in [`agents/CONTEXT.md`](../CONTEXT.md), so **Loop** means the per-program setting.

## Summary

The app-wide Loop setting is removed. Each program sets its own Loop with `#{loop_after_beats=N}`. A pending launch of a program with this annotation plays it again every N beats until it is stopped. A program without it plays once. Loop can be edited live with `Cmd-N l` on the keyboard, or by holding Loop on the Launchkey. Both write the annotation back to the source. The gutter and the selection message show it.

## 1. Removing the app-wide setting

*Decided while charting.*

Remove all of these:

- `AppState.repeat_after_measures`, and every `repeat_after_measures` argument that carries its value (in `actions.rs`, `effects.rs` and `player.rs`).
- `Action::CycleRepeatAfterMeasures`, which the Function button maps to (`Event::PadFunctionDown` in `midi_input.rs`), and the Function pad colours that show its state.

Nothing replaces them as a global default, and there's no migration because the setting was never saved.

## 2. Syntax and checking

*Decided while charting. The range comes from [Which prefixed commands edit Loop from the keyboard?](tickets/04-keyboard-loop-commands.md)*

- The annotation is spelled `loop_after_beats=N`, where N is a positive whole number with no upper bound, e.g. `#{loop_after_beats=4}` or `#{color=…, loop_after_beats=12}`. It becomes a new `expr::Annotation` variant beside `Keys`, and its `Display` round-trips.
- `0`, negative numbers, fractions (`1.5`) and non-numbers are errors. Fractional beats are out of scope.
- `loop_after_beats` together with `keys` on the same binding is an error.
- Like any annotation, it gives the binding a program slot.

## 3. Which launches loop

*Decided while charting.*

Today's split stays the same, except that N now comes from the program's annotation:

| Launch | Starts | Loops? |
| --- | --- | --- |
| Bottom pad, Play button, evaluate-and-play | next measure (pending) | yes, if the program has `loop_after_beats` |
| Top pad, `ToggleProgramPlayback` | immediately | never: it plays once, even with the annotation |

**Open:** the keyboard launches that pick their own period are still undecided: Cmd+Return, Shift+Cmd+Return, and plain Return in Edit mode. See [What do Cmd+Enter and Shift+Cmd+Enter do once Loop is per program?](tickets/07-keyboard-launch-periods.md). Until that ticket is resolved, this spec isn't ready to implement.

## 4. Timing

*[How do loops that aren't a whole number of measures behave?](tickets/02-loops-off-the-measure.md)*

- A looping program plays again every N beats, counted from the measure boundary where it was launched. The same rule covers waveform and sequence programs.
- **Drift is allowed.** Nothing snaps the program back to the barline, and nothing rounds N. In 4/4, `loop_after_beats=3` lands on beats 1, 4, 3, 2, and comes back to the barline every 12 beats.
- **Plays overlap.** If a play is still sounding when the next one starts, both sound. Nothing is cut off and nothing warns. This is the tracker's behaviour today.
- **Sequences play as written.** Every pass plays all of its steps, even when they extend past beat N, so passes can overlap. No steps are dropped and the program isn't rejected.
- Internally the period changes from measures to beats. `Player`'s `repeat_after_measures` parameters, and the `measures * beats_per_measure` conversion inside them, become a period in beats.
- Tempo and beats per measure stay fixed for the session. Changing them live is out of scope.

## 5. Editing a program that is already looping

*[When does a changed Loop take effect on a program that is already looping?](tickets/03-editing-a-looping-program.md)*

- A Loop change takes effect at the program's **next launch**, whether it adds, changes or removes the Loop, and whether it came from the keyboard, the Launchkey or an edit to the file. Whatever is already sounding or queued keeps its period. A reload already stops every voice and clears every queued play.
- **Step edits on a sequence** (`Player::play_step`):
  - While a sequence is looping, a newly enabled step takes the period that is already running, as it does today.
  - Before the first pass starts, it takes the program's own `loop_after_beats`, or plays once without it. This replaces the `default_repeat_measures` argument, which comes from the app-wide setting today.
- No guard against stacked loops is needed. Entering Edit mode already removes the program's queued plays, and Play and the bottom pad already refuse to queue when something is queued. Both stay as they are.

## 6. Keyboard: `Cmd-N l`

*[Which prefixed commands edit Loop from the keyboard?](tickets/04-keyboard-loop-commands.md)*

- The `Cmd-N` prompt (`PrefixKey::Annotation`) becomes "Annotation? (k)eys (u)p (d)own (l)oop".
- `l` starts a new prefix key, `PrefixKey::Loop`, which prompts "Loop kick? (1-9) beats (0) once".
- The next key finishes the prefix:

  | Key | Writes | Message |
  | --- | --- | --- |
  | `1`–`9` | `loop_after_beats=N` | "kick loops every N beats" |
  | `0` | removes `loop_after_beats` | "kick plays once" |
  | anything else | nothing | cancels, as any prefix does |

  A modifier press or a held key's auto-repeat neither finishes nor cancels the prefix.
- If the program is playing, the message adds " from its next launch", e.g. "kick loops every 3 beats from its next launch" (§5).
- It works in the same editor modes as `Cmd-N`: Select and Keys. In Keys mode, the key that finishes the prefix plays no note.
- These checks happen when `l` is pressed, and if one fails no Loop prefix starts:
  - an empty slot: "B:3 is empty"
  - stale source: the existing stale-source message
  - a `#{keys}` program: "kick plays keys, so it can't loop"
- `Cmd-N k` on a program that has a Loop is refused with "Remove kick's Loop first".
- The value is written back to the source the same way `k` writes `#{keys}`. When the binding loses its only annotation, `level_db=…` takes its place, as when `keys` is removed.
- Every message names the program through `display_name`, so an unnamed program reads "B:3 loops every 8 beats".

## 7. Launchkey: holding Loop

*Hardware facts come from [Which Launchkey control should carry the Loop prefix, and what can we do with it?](tickets/01-launchkey-loop-button.md). Behaviour comes from [How do the Launchkey's Loop prefix and pads display and set Loop?](tickets/05-launchkey-loop-prefix.md). The [pad-layout prototype](assets/05-launchkey-loop-prefix-prototype.html) shows variant B.*

- **Input.** The Loop button sends CC 118 on channel 1 of the DAW port: value 127 when pressed and 0 when released (confirmed on the hardware). `launchkey.rs` gains a press event and a release event for it.
- **Holding Loop opens a temporary DAW pad sub-mode, Loop.** While Loop is held, the pads only edit the selected program's Loop, whichever sub-mode they were in before. Releasing Loop returns them to that sub-mode.
- **Layout.**
  - The top row (notes 96–103) is 1–8 beats.
  - The bottom row (notes 112–119) is 1–8 measures, that is `k × beats_per_measure` beats, so 4–32 beats in 4/4.
- **Lighting.**
  - Every pad whose value equals the picked value is white. In 4/4, a Loop of 4 or 8 beats lights two pads, one in each row.
  - Every other pad is lit dimly in the program's colour.
  - When the picked value is "plays once", no pad is white.
- **Pressing pads.**
  - Each press picks that pad's value, and the last press wins.
  - Pressing a white pad picks "plays once".
  - The picked value is held in memory, and the pads update straight away.
- **Writing.** Releasing Loop writes the picked value to the source and shows the same message as `Cmd-N l`, including " from its next launch" when the program is playing. If no pad was pressed, the source isn't changed.
- **Device screen.** While Loop is held, the device screen shows the program and the picked value, e.g. "kick" / "Loop: 10 beats", or "Loop: once". It updates on every press, so a value with no pad of its own, such as 10 beats, can still be read.
- **Refusals.** The same rules as `Cmd-N l` (§6) apply:
  - only in the Select and Keys editor modes, so never on an owned program
  - not on an empty slot, stale source or a keys program
  - checked when Loop is pressed and again when it is released

  A refused program gets dark pads while Loop is held, and the reason appears on the device screen and the status line, e.g. "lead plays keys, so it can't loop".
- **Other controls** keep their normal jobs while Loop is held.
  - If Track ◀ ▶ changes the selection, any picked value is dropped and the pads show the newly selected program.
  - Leaving the DAW pad layout (Shift + pads) while Loop is held drops the picked value and writes nothing.
- **Loop button light.** It is lit when the selected program has a Loop, and dark when it plays once. The brightness is set by CC 118 on channel 4, the way tuun drives Record's light. To verify on the hardware: the light is single-colour and responds to brightness through that CC.
- **Function button** (CC 105) does nothing, and its light stays off.

## 8. Screen

*[How does tuun's screen show a program's Loop?](tickets/06-loop-on-screen.md)*

- **Gutter.** This is the prompt column, where the active program shows ` ▸ `.
  - A program that isn't active shows `↻` if it has `loop_after_beats`, or `⌨` if it has `keys`, drawn in the program's colour.
  - The gutter never shows N.
  - The active program keeps today's prompt whatever its annotations: ` ▸ ` in Select, Move Sliders and Keys modes, and blank in Edit and Record modes.
- **The glyphs follow the annotation, not playback.** `↻` appears or disappears as soon as the source changes, even while an old loop is still sounding. `⌨` marks the annotation, whether or not the instrument is installed.
- **Selection message.** Selecting a program sets the status to its bank-relative address, then its binding name, then its Loop or keys setting:
  - `B:3 kick, loops every 4 beats`
  - `B:3 lead, plays keys`
  - `B:3 kick` (no Loop, not keys)
  - `B:3` (anonymous `_` binding)
  - no text for an empty slot, as today

  A new `ProgramSet` helper beside `display_name` builds this message. The message stays under the status line's 45 characters.
- **Launch messages** keep today's wording, with N from the annotation, always in beats:
  - Pending and looping: "Looping waveform kick every 4 beats", "Looping sequence kick every 6 beats"
  - Pending, plays once: "Playing waveform kick", "Playing sequence kick"
  - Immediate: "Playing voice of kick", even for a program that has a Loop
- **While Loop is held on the Launchkey**, the status line mirrors the device screen, e.g. "Loop kick: 10 beats" or "Loop kick: once", and updates on every press. Releasing Loop replaces it with the write message. The gutter keeps showing the written annotation until the release.
- **Units** are beats everywhere.
- Hypothesis (unverified): the renderer's font has both `↻` (U+21BB) and `⌨` (U+2328). Check this on the real font during implementation.

## 9. Documentation

*Decided while charting.*

- **README, Launchkey section.**
  - Remove the bullets saying bottom pads "optionally repeat every measure or every _other_ measure" and that the "Function" button switches between repeat behaviours.
  - Say that bottom pads and Play loop according to the program's `loop_after_beats`.
  - Describe holding Loop.
- **README, Keyboard Navigation.** Add "Cmd + N, then L" in Select and Keys modes. Update the Cmd + Enter and Shift + Cmd + Enter lines once the [keyboard launch ticket](tickets/07-keyboard-launch-periods.md) is resolved.
- **`docs/language-spec.md` § Annotations.** Add `loop_after_beats=N`: its range, that it can't be combined with `keys`, and which launches honour it.
- **`agents/CONTEXT.md`.** The Loop, Gutter and DAW pad sub-mode entries already match this spec, so they need no change.

## Out of scope

These are listed on the [map](map.md#out-of-scope):

- fractional beats
- an app-wide default Loop
- looping immediate launches
- changing tempo or beats per measure live
