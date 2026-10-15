# Measure the Edit-mode helpers

Labels: wayfinder:task
Status: open
Assignee: djspoons
Blocked by: none

## Question

How long does each Edit-mode helper hold the main thread, on the dev's machine?
- completion (`apply_complete` in `src/lib/actions.rs`) and parameter hints (`apply_parameter_hint`), which evaluate every binding before the program;
- ShowType (`apply_show_type`);
- the parse after each keystroke (`parse_program_errors`, via `edit_text_op`).

Measure each on a program near the start and near the end of a large source file (such as `large.tuun` in [research/processing-timing](../research/processing-timing/README.md)), in a debug and a release build. `keyboard input` under `--show-loop-timing` includes these helpers but has SDL's 1 ms resolution and doesn't name the helper, so add a timing for each helper.

Done when the numbers are recorded in this ticket's resolution. They decide whether the helpers stay on the main thread, as [Where program processing runs](06-evaluation.md) left them, or need caching or a move.

## Instrumentation

`--show-loop-timing` now also shows three lines, between `render present` and `text textures`, each as `current second (10 s max)` in ms like the other lines:
- `complete`: one Complete keystroke (Cmd-/), which is completion after an identifier and a parameter hint after `(`.
- `show type`: one ShowType keystroke (Ctrl-H).
- `text edit`: one keystroke that inserts or deletes text (typing, Backspace, Delete, word deletes, Ctrl-K), which re-parses the program.

Each covers the whole dispatch of that keystroke's action, effects included, timed with `Instant` rather than SDL's 1 ms ticks. Completion and parameter hints share a line, so measure them one at a time.

Code: `Helper` and `LoopTiming::record_helper` in `src/lib/loop_timing.rs`; `edit_helper` and the timed dispatch in the SDL event loop in `src/main.rs`. The `CLAUDE.md` checks pass. It has not been run with a display.

## Checklist for the dev

Run with `--show-loop-timing` on `large.tuun` from [research/processing-timing](../research/processing-timing/README.md), once in a debug build and once in a release build (`--release`). In each build, for a program near the start of the file and one near the end:

1. Enter Edit mode, wait for the maximums to clear (10 s), then type a few characters and delete them. Record `text edit`.
2. After an identifier fragment (e.g. `si`), press Cmd-/ a few times. Record `complete` as completion.
3. Wait 10 s. After the name of a function and `(`, press Cmd-/ once. Record `complete` as parameter hint.
4. With the cursor in a name, press Ctrl-H. Record `show type`.

Fill in the table, each cell `current second / 10 s max`:

| Program | Build | completion | parameter hint | show type | text edit |
|---|---|---|---|---|---|
| near start | debug | | | | |
| near end | debug | | | | |
| near start | release | | | | |
| near end | release | | | | |
