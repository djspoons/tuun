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

*Decided while charting, and in [What do Cmd+Enter and Shift+Cmd+Enter do once Loop is per program?](tickets/07-keyboard-launch-periods.md)*

Every launch either starts at the next measure and follows the program's Loop, or starts now and plays once. No key or pad picks its own period.

| Launch | Starts | Loops? |
| --- | --- | --- |
| Bottom pad, Launchkey Play, Edit-mode Return, finishing a recording with Play | next measure (a recording: the take's end boundary) | as the program's `loop_after_beats` says, or once without it |
| Top pad (Toggle or Trigger), Cmd+Return in Select or Edit mode | now | never, even with the annotation |

- **Cmd+Return**, in Select and Edit modes, plays the whole program once, starting now. Edit mode evaluates first.
  - For a sequence, the pattern's first beat lands at the press.
  - It never stops what is already sounding, whatever the launch mode, so pressing it twice layers two plays.
  - On a program that can't play, it shows today's "Program B:3 is not a valid waveform" message.
- **Return in Edit mode** evaluates, then makes a pending launch. In Select mode, Return still enters Edit mode.
- **Shift+Cmd+Return** is removed in both modes, so the key does nothing.
- **Finishing a recording with Play** (`finish_take`) passes the slot's `loop_after_beats` where it passes `state.repeat_after_measures` today.
- **A Trigger-mode top pad** plays the whole program from now, like Cmd+Return, so on a sequence it plays the pattern rather than one hit. Delete `Action::StartProgramVoice`, `Effect::PlayProgramVoice` and `Player::play_program_voice`, and route Trigger through `play_program(…, Start::Now, None)` without the stop that Toggle does.
- `Action::PlayProgram` and `Action::EvaluateAndLeaveEditMode` lose their `repeat_after_measures` field. The pending or immediate choice decides whether the annotation is read.

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
- **Loop button light.** It is bright (127) when the selected program has a Loop, and at its lowest level (0) when the program plays once. The brightness is set by CC 118 on channel 4, the way tuun drives Record's light.
  - Verified on the hardware: the light is white only.
  - Values from 0 to 127 run from lowest to bright. 0 doesn't turn it off: it stays at the same faint level as the Stop button next to it, so "plays once" looks dim, not dark. All the transport buttons, Undo and Metronome behave this way. Using the full scale from 0 to 127 is intended.
  - Values sent on channel 1 also change only its brightness.
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
  - Immediate (top pads and Cmd+Return): "Playing waveform kick", "Playing sequence kick", even for a program that has a Loop. "Playing voice of kick" goes away with `play_program_voice` ([ticket 07](tickets/07-keyboard-launch-periods.md)).
- **While Loop is held on the Launchkey**, the status line mirrors the device screen, e.g. "Loop kick: 10 beats" or "Loop kick: once", and updates on every press. Releasing Loop replaces it with the write message. The gutter keeps showing the written annotation until the release.
- **Units** are beats everywhere.
- The renderer's font (Arial Unicode) has both `↻` (U+21BB) and `⌨` (U+2328). This was verified on the real font.

## 9. Documentation

*Decided while charting.*

- **README, Launchkey section.**
  - Remove the bullets saying bottom pads "optionally repeat every measure or every _other_ measure" and that the "Function" button switches between repeat behaviours.
  - Say that bottom pads and Play loop according to the program's `loop_after_beats`.
  - Describe holding Loop.
  - Remove "For sequenced waveforms, 'Trigger' mode only plays one step instead of the whole sequence".
- **README, Keyboard Navigation.**
  - Add "Cmd + N, then L" in Select and Keys modes.
  - Select mode: Cmd + Enter plays the current program once, now.
  - Edit mode: Enter plays at the next measure, looping as its `loop_after_beats` says, and Cmd + Enter plays it once, now.
  - Remove both Shift + Cmd + Enter lines.
- **`docs/language-spec.md` § Annotations.** Add `loop_after_beats=N`: its range, that it can't be combined with `keys`, and which launches honour it.
- **`agents/CONTEXT.md`.** The Loop, Gutter, DAW pad sub-mode and Trigger entries already match this spec, so they need no change.

## Implementation phases

The sections above are grouped by topic, not by change, so they don't work as phases on their own:

- §1 removes the only source of a period before anything replaces it, so the `repeat_after_measures` plumbing would be deleted and then re-added in beats.
- §3, §4, §5 and §8's launch messages all edit the same code (`play_program_effects`, `Player::play_waveform`, `Player::play_sequence`, `Player::play_step`, `play_waveform_message`). Done one section at a time, those lines would change three or four times.
- §8 is two changes. The gutter and selection message need only §2, but the launch messages need §3.
- §7 reuses §6's refusal checks and write-back.
- §9's documentation should go with the phase that changes the behaviour it describes.

Each phase below is one change that reads on its own. Phase 4 can go anywhere after phase 1. The other phases are in dependency order.

### Phase 1: the annotation (§2)

- Add the `loop_after_beats` `expr::Annotation` variant, with its parsing, range errors, the error for combining it with `keys`, and a `Display` that round-trips.
- Expose it on `programs::Program` the way `keys` and `level_db` are exposed.
- Nothing reads it yet.
- Docs: `docs/language-spec.md` § Annotations, for the syntax, range and `keys` rule. Which launches honour it waits for phase 3.

### Phase 2: immediate launches play once, now (§3's "now" row)

After this phase, every launch either starts now and plays once, or starts at the next measure with the app-wide setting's period. Phase 3 then only has to change where the period comes from, in one place.

- Trigger-mode top pads go through `play_program(…, Start::Now, None)`. Delete `Action::StartProgramVoice`, `Effect::PlayProgramVoice`, `Player::play_program_voice` and their tests. Move the useful test cases (layering, no stop) over to the new route.
- Cmd+Return in Select and Edit modes plays once, now. Return in Edit mode makes a pending launch. Shift+Cmd+Return is removed, so the `if shift { 2 } else { 1 }` arms in `sdl2_input.rs` go.
- `Action::PlayProgram` and `Action::EvaluateAndLeaveEditMode` replace their `repeat_after_measures` field with a pending-or-now choice. A pending launch reads `state.repeat_after_measures`, as the bottom pad does today. Check whether `start_at_next_measure` then says the same thing and can go.
- Docs: README Keyboard Navigation (the Cmd+Enter and Enter lines, and removing Shift + Cmd + Enter), and removing the README's Trigger-mode "only plays one step" line.

### Phase 3: per-program Loop replaces the app-wide setting (§1, §4, §5, §8's launch messages)

- Remove `AppState.repeat_after_measures`, `Action::CycleRepeatAfterMeasures`, the Function pad colours in `midi_input.rs` and the `PadFunctionDown` mapping. If nothing else uses `Event::PadFunctionDown` or the Function light, remove them from `launchkey.rs` too, so the light stays off (§7).
- A pending launch and `finish_take` read the slot's `loop_after_beats`.
- `Player`'s period parameters become beats. Remove the `measures * beats_per_measure` conversions, and the `beats_per_measure` argument of `play_waveform_message` if nothing else needs it. Give the waveform and sequence launch messages one helper, so they word "every N beats" the same way. Today the sequence message formats N with `{:?}`.
- `play_step`'s `default_repeat_measures` becomes the program's own `loop_after_beats`. Since `Player` already has the `ProgramSet`, it can read the annotation itself instead of taking an argument.
- Add tests for drift (`loop_after_beats=3` in 4/4) and for a sequence that runs past beat N.
- Docs: README Launchkey (remove the "optionally repeat" and Function bullets, and say that bottom pads and Play follow `loop_after_beats`), and the "which launches honour it" part of `docs/language-spec.md`.

### Phase 4: the screen (§8's gutter and selection message)

- Gutter glyphs `↻` and `⌨`.
- The `ProgramSet` selection-message helper beside `display_name`, used wherever selection sets the status today.

### Phase 5: `Cmd-N l` (§6)

- `PrefixKey::Loop`, the prompts, and the finishing keys.
- Put the refusal checks (empty slot, stale source, keys program) in one function that returns the refusal message, because phase 6 calls it again. Add the matching "Remove kick's Loop first" refusal to `k`.
- Make the source write-back that `k` uses take any annotation, rather than copying it for `loop_after_beats`. This includes `annotation_edits` and the `level_db=…` stand-in in `programs.rs`.
- Put the "loops every N beats" / "plays once" / " from its next launch" message in one function, because phase 6 shows the same message.
- Docs: README "Cmd + N, then L".

### Phase 6: holding Loop on the Launchkey (§7)

- The CC 118 press and release events in `launchkey.rs`, the Loop pad sub-mode with its layout and lighting, the picked value, the device screen, the status-line mirror, and the Loop button light.
- On press and on release, use phase 5's refusal function. On release, use phase 5's write-back and message.
- Docs: README Launchkey, "holding Loop".

## Out of scope

These are listed on the [map](map.md#out-of-scope):

- fractional beats
- an app-wide default Loop
- looping immediate launches
- changing tempo or beats per measure live
