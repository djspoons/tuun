# Measure on the dev's machine

Labels: wayfinder:task
Status: open
Assignee: djspoons
Blocked by: [Instrument the main loop](01-instrument-main-loop.md)

## Question

Run the instrumented build and record each step's time and the input latency for each input path in these scenarios:

- idle;
- several programs playing;
- holding chords on a keys instrument;
- a large source file loaded;
- entering or reloading a program while notes play.

Done when the numbers are recorded in this ticket's resolution.

## Results so far

Each value is `current second / 10 s max`, in ms.

| Step or input | Idle | Idle, switching apps | Several programs playing | Source reload | Enter a program while others play |
|---|---|---|---|---|---|
| SDL events | 0.0 / 0.2 | 0.0 / 12.7 | 0.0 / 9.2 | 0.0 / 2585.0 | 0.0 / 233.5 |
| Launchkey events | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| status wait | 11.0 / 11.0 | 11.0 / 11.1 | 11.0 / 11.1 | 11.0 / 11.1 | 11.0 / 11.1 |
| tick | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| render | 5.0 / 5.5 | 4.9 / 44.5 | 5.2 / 7.1 | 5.0 / 6.8 | 4.9 / 6.7 |
| Launchkey update | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| keyboard input | 0.0 / 0.0 | 0.0 / 1.0 | 0.0 / 0.0 | 0.0 / 2585.0 | 0.0 / 233.0 |
| pad input | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| piano key input | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

Notes:
- **Source reload** blocks the main thread for about 2.6 s inside the SDL events step: the reload key's handling took 2585 ms. Any pad, piano key or typed key arriving during that time waits for it.
- **Several programs playing:** one SDL events pass took 9.2 ms, but no keyboard input above 0 was recorded. Until the fix noted in [Instrument the main loop](01-instrument-main-loop.md), keyboard input counted only key-downs, not typed characters (text input); the 9.2 ms may have come from one of those or from another event.
- **Entering a program** (editing it, then leaving Edit mode with Return) while other programs play blocks the main thread for about 233 ms in the SDL events step.
- **A large source file:** the source reload column was measured on a large file with many dependencies, including `synth.sub`, so it also covers this scenario.
- **Pads and Launchkey piano keys** are not measured yet: the Launchkey hardware isn't available for now.
