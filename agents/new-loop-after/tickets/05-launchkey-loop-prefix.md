---
type: prototype
status: closed
assignee: djspoons
blocked_by: [01]
---

# How do the Launchkey's Loop prefix and pads display and set Loop?

## Question

With a Launchkey button acting as the Loop prefix, how do the pads show the active (or a chosen) program's current Loop, and how does pressing a pad set it? Which values do the pads map to (e.g. 1–16 beats, or beats in the top row and measures in the bottom), and how is "no loop" chosen? Does the prefix target the selected program or the next pad pressed? How is a value the pads can't show displayed? What does that button's LED show at rest? First settle which button is the prefix: Loop (CC 118, white LED) or Function (CC 105, RGB; it is freed when the app-wide Loop goes). Loop's hardware label now matches the glossary term. The keyboard sets 1–9 beats and the source has no upper bound ([Which prefixed commands edit Loop from the keyboard?](04-keyboard-loop-commands.md)), so the pads need some way to show a value they can't set. Prototype the pad layout so it can be reacted to.

## Assets

- Prototype: [assets/05-launchkey-loop-prefix-prototype.html](../assets/05-launchkey-loop-prefix-prototype.html), a simulated Launchkey with three layouts. Open it in a browser and switch layouts with `?variant=A|B|C` or the bar at the bottom.

## Resolution

Of the prototype's three layouts, the pads use variant B (beats on top, measures below). There is no prefix: the editor is active only while the **Loop** button is held. The Function button is freed.

- **Holding Loop is a temporary DAW pad sub-mode.** While Loop (CC 118) is held, the pads only edit the selected program's Loop, whatever sub-mode the pads were in before. Releasing Loop puts back the earlier sub-mode. Confirmed on the hardware: Loop sends CC 118 on channel 1 with value 127 on press and 0 on release.
- **Layout.** The top row is 1–8 beats. The bottom row is 1–8 measures, which is 4–32 beats in 4/4.
- **Lighting.** Every pad whose value equals the selected program's Loop is lit white. In 4/4, 4 and 8 beats light two pads each, one in each row. Every other pad is lit dimly in the program's colour. When the program plays once, no pad is white.
- **Pressing pads.** A pad press picks that value, and the last press wins. Pressing a white pad again picks "plays once", so every pad goes dim. The value is held in memory and the pads update straight away.
- **Writing the value.** Releasing Loop writes the picked value to the source. It shows the same message as `Cmd-N l` ("kick loops every N beats", "kick plays once", with "from its next launch" added while the program plays). Releasing without pressing a pad leaves the source unchanged.
- **Device screen.** While Loop is held, the screen shows the program and the value picked so far (e.g. "kick" / "Loop: 10 beats", or "Loop: once"). It updates on every pad press, so a value with no pad (10 beats, 40 beats) can still be read.
- **Refusals.** These are the same as for `Cmd-N l`: the same editor modes (Select and Keys only, so not an owned program), empty slots, out-of-date source, and keys programs. They are checked when Loop is pressed and again on release. A refused program gets dark pads while Loop is held, and the reason goes on the device screen and the status line ("lead plays keys, so it can't loop").
- **Other controls** do their normal job while Loop is held. If Track ◀ ▶ changes the selection, any value already picked is dropped, and the pads show the newly selected program. Leaving the DAW pad layout (Shift + pads) while Loop is held also drops the picked value and writes nothing.
- **The Loop button's light.** It is lit when the selected program has a Loop and dark when it plays once (brightness through CC 118 on channel 4).
- **Function** (CC 105) is freed: it has no action and its light stays off.

Rejected: variant A (a 1–16 beat ruler) and variant C (holding Function to cycle every program in the bank through set values), and a press-then-pad prefix.
