# Where completion and parameter hints run

Labels: wayfinder:grilling
Status: open
Assignee: djspoons
Blocked by: none

## Question

Completion and parameter hints (Cmd-/ in Edit mode, `apply_complete` and `apply_parameter_hint` in `src/lib/actions.rs`) block the main thread for 69–317 ms in release and 193–920 ms in debug on `large.tuun`, growing with the program's position in the file ([Measure the Edit-mode helpers](09-measure-edit-helpers.md)). Both evaluate every binding before the program. Should they move to the processing thread decided in [Where program processing runs](06-evaluation.md), should the main thread keep the bindings' results between keystrokes, or something else? What does the user see while a completion is pending?
