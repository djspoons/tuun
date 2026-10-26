---
type: prototype
status: open
assignee: djspoons
blocked_by: [01]
---

# How do the Launchkey's Loop prefix and pads display and set Loop?

## Question

With a Launchkey button acting as the Loop prefix, how do the pads show the active (or a chosen) program's current Loop, and how does pressing a pad set it? Which values do the pads map to (e.g. 1–16 beats, or beats in the top row and measures in the bottom), and how is "no loop" chosen? Does the prefix target the selected program or the next pad pressed? How is a value the pads can't show displayed? What does that button's LED show at rest? First settle which button is the prefix: Loop (CC 118, white LED) or Function (CC 105, RGB; it is freed when the app-wide Loop goes). Loop's hardware label now matches the glossary term. The keyboard sets 1–9 beats and the source has no upper bound ([Which prefixed commands edit Loop from the keyboard?](04-keyboard-loop-commands.md)), so the pads need some way to show a value they can't set. Prototype the pad layout so it can be reacted to.

## Assets

- Prototype: [assets/05-launchkey-loop-prefix-prototype.html](../assets/05-launchkey-loop-prefix-prototype.html), a simulated Launchkey with three layouts. Open it in a browser and switch layouts with `?variant=A|B|C` or the bar at the bottom.
