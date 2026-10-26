---
type: grilling
status: closed
assignee: djspoons
blocked_by: []
---

# How does tuun's screen show a program's Loop?

## Question

Where and how does the screen show each program's Loop? Does the program list show it, and in what form ("4 beats", "1 measure", "⟳4")? Does the status line show it for the selected program? What wording do launch messages use, replacing today's "Looping waveform B:3 every 4 beats" for pending launches that loop and one-shot launches? The edit messages are already settled ([Which prefixed commands edit Loop from the keyboard?](04-keyboard-loop-commands.md), [When does a changed Loop take effect on a program that is already looping?](03-editing-a-looping-program.md)). The open question is whether the screen also shows a pending Loop edit while Loop is held on the Launchkey ([How do the Launchkey's Loop prefix and pads display and set Loop?](05-launchkey-loop-prefix.md)). It also needs to settle how a program that is still looping with its old period is shown after its Loop changed "from its next launch".

## Resolution

The screen shows Loop in two places: a glyph in the gutter for programs other than the active one, and the message shown when a program is selected. Nothing on the status line stays up persistently.

- **Gutter.** A non-active program with `loop_after_beats` shows `↻` in the prompt column (where the active program's ` ▸ ` goes), in the program's colour. A non-active `#{keys}` program shows `⌨` there in the same way. No row shows N. The active program keeps today's prompt: ` ▸ ` in Select, Move Sliders and Keys modes, and blank in Edit and Record modes, whatever its annotations. So the column never has to show both the selection and a setting, and a program with no `color=` can't be mistaken for the selection.
- **The glyphs follow the annotation.** They don't follow playback or installation. Adding Loop shows `↻` at once, and removing Loop clears it at once, even while the old loop still sounds "until its next launch" ([When does a changed Loop take effect on a program that is already looping?](03-editing-a-looping-program.md)). `⌨` marks the annotation, not whether the instrument is installed.
- **Selection message.** Selecting a program replaces the status with the bank-relative address, then the binding name, then its Loop or keys setting. A program can't have both ([Which prefixed commands edit Loop from the keyboard?](04-keyboard-loop-commands.md)), so the message stays under the status line's 45 characters:
  - `B:3 kick, loops every 4 beats`
  - `B:3 lead, plays keys`
  - `B:3 kick` (no Loop, not keys)
  - `B:3` (anonymous `_` binding)
  - no text for an empty slot, as today

  The message deliberately shows both the address and the name. It goes through a new `ProgramSet` helper beside `display_name`, so no raw index is formatted by hand.
- **Launch messages.** Today's wording stays. N comes from the annotation and is always given in beats: "Looping waveform kick every 4 beats" and "Looping sequence kick every 6 beats" for pending launches that loop, and "Playing waveform kick" and "Playing sequence kick" for pending launches that play once. Immediate launches stay "Playing voice of kick", even for a program that has a Loop.
- **Units.** Beats everywhere, matching the annotation, the `Cmd-N l` digits and the edit messages.
- **Holding Loop on the Launchkey.** The status line mirrors the device screen ([How do the Launchkey's Loop prefix and pads display and set Loop?](05-launchkey-loop-prefix.md)): "Loop kick: 10 beats" or "Loop kick: once", updated on every pad press. Releasing Loop replaces it with the write message. The gutter keeps showing the written annotation until then.
- **Glyph font.** Hypothesis (unverified): the renderer's font (Arial Unicode) has both `↻` (U+21BB) and `⌨` (U+2328), since both are Unicode 1.1. The implementation should check this on the real font.

Considered and dropped: a badge at the right edge of the row; `↻N` in the gutter (N has no upper bound and the column is about 3 characters wide); replacing the active program's triangle with `↻` (it collides with the selection, and with programs that have no colour); showing the running period after a "from its next launch" change; a persistent Loop readout on the status line.
