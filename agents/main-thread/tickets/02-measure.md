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

| Step or input | Idle |
|---|---|
| SDL events | 0.0 / 0.2 |
| Launchkey events | 0.0 / 0.0 |
| status wait | 11.0 / 11.0 |
| tick | 0.0 / 0.0 |
| render | 5.0 / 5.5 |
| Launchkey update | 0.0 / 0.0 |
| keyboard input | 0.0 / 0.0 |
| pad input | 0.0 / 0.0 |
| piano key input | 0.0 / 0.0 |

