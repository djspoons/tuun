# Measure the Edit-mode helpers

Labels: wayfinder:task
Status: open
Assignee:
Blocked by: none

## Question

How long does each Edit-mode helper hold the main thread, on the dev's machine?
- completion (`apply_complete` in `src/lib/actions.rs`) and parameter hints (`apply_parameter_hint`), which evaluate every binding before the program;
- ShowType (`apply_show_type`);
- the parse after each keystroke (`parse_program_errors`, via `edit_text_op`).

Measure each on a program near the start and near the end of a large source file (such as `large.tuun` in [research/processing-timing](../research/processing-timing/README.md)), in a debug and a release build. `keyboard input` under `--show-loop-timing` includes these helpers but has SDL's 1 ms resolution and doesn't name the helper, so add a timing for each helper.

Done when the numbers are recorded in this ticket's resolution. They decide whether the helpers stay on the main thread, as [Where program processing runs](06-evaluation.md) left them, or need caching or a move.
