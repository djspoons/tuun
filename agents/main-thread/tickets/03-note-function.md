# Where the note function runs

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

Should evaluating the keys instrument's note function (`environment.apply_note_function`, called for every note-on in `src/lib/effects.rs`) stay on the main thread, or should note handling move off it?

## Resolution

Scope: live note-on and note-off, from the computer keyboard in Keys mode and from Launchkey piano keys. A program that plays a phrase through a keys instrument evaluates the note function during program processing, so it falls under [Where program processing runs](06-evaluation.md).

**Decisions**

1. **Stay.** Note-on keeps evaluating the note function on the main thread.
2. **Budget.** About 5 ms per main-loop pass, counting every note-on handled in that pass (a chord), not each note alone.
3. **Never behind processing.** If note handling ever moves, it does not share the processing thread, which can be busy for seconds during a reload.
4. **No install-time precompute.** Velocity stays continuous; a cheaper note-on is the remedy if one is needed, not a lookup table.
5. **Decided without Launchkey numbers.** Piano keys reach the same `PlayNoteOn` effect as the computer keyboard, so the note function costs the same. Their extra latency is the status wait: [The 10 ms status wait](04-status-wait.md).
6. **Cheaper note-ons are code TODOs, not tickets.** One in `Environment::apply_note_function` (`src/lib/environment.rs`), on the cloning in `eval::substitute`. One in `PlayNoteOn` (`src/lib/effects.rs`): substitute only the sliders the instrument uses.

**Measurements behind these decisions**

Timed outside the UI on a 4-core Linux container, not the dev's machine, so only proportions transfer. 200 notes per instrument, keys 36–96, a spread of velocities, using the stand-in files in [research/processing-timing](../research/processing-timing/README.md) and `lib/v0/lessons/counterpoint/first.tuun`. Times in ms.

| Instrument | Build | `apply_note_function` min / median / p99 / max | Slider substitution median / p99 |
|---|---|---|---|
| `pm.brass_keys` (large.tuun) | release | 0.09 / 0.11 / 0.15 / 0.18 | 0.08 / 0.13 |
| `sub.flute_keys` (medium.tuun) | release | 0.61 / 0.68 / 1.16 / 3.78 | 0.09 / 0.16 |
| `sub.flute_keys` (large.tuun) | release | 0.72 / 0.92 / 1.23 / 1.36 | 0.14 / 0.23 |
| `pm.brass_keys` (large.tuun) | debug | 0.32 / 0.43 / 0.79 / 2.37 | 0.83 / 1.38 |
| `sub.flute_keys` (large.tuun) | debug | 2.19 / 2.39 / 3.14 / 3.21 | 0.80 / 1.50 |

- No module code is evaluated again per note: the instrument's free variables were filled in when it was evaluated. The cost is cloning and rebuilding the function body for its arguments, and again for each nested call; optimizing the result is about 0.01–0.02 ms.
- Slider substitution walks the note-on once for every slider in every program, so it costs (all sliders) × (note-on size).
- `play_note` is one send on an unbounded channel.
- On the dev's machine, a note-on in Keys mode took at most about 1 ms ([Measure on the dev's machine](02-measure.md)).
- **Near the budget:** a four-note flute chord in a debug build is about 8–10 ms in one pass (about 3 ms in release). The README runs the UI as a debug build.
