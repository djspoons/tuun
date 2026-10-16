# Where completion and parameter hints run

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Completion and parameter hints (Cmd-/ in Edit mode, `apply_complete` and `apply_parameter_hint` in `src/lib/actions.rs`) block the main thread for 69–317 ms in release and 193–920 ms in debug on `large.tuun`, growing with the program's position in the file ([Measure the Edit-mode helpers](09-measure-edit-helpers.md)). Both evaluate every binding before the program. Should they move to the processing thread decided in [Where program processing runs](06-evaluation.md), should the main thread keep the bindings' results between keystrokes, or something else? What does the user see while a completion is pending?

## Resolution

**What the helpers evaluate.** Completion and parameter hints both call `Environment::program_context`, which evaluates `ProgramSet::evaluation_bindings(i)`: the file-level bindings before program *i*, each earlier definition wrapped in its own slider values, and program *i*'s own sliders. None of these read program *i*'s text (slider configs come from the binding's annotations), so editing in Edit mode never changes the context; only slider values and a source reload do. A completion cycle (repeated Cmd-/) already reuses its candidates without evaluating again.

**Decisions**

1. **Prefetch the context on the processing thread.** Entering Edit mode on program *i* queues a job on the processing thread from [Where program processing runs](06-evaluation.md) that evaluates program *i*'s context. The reply is stored on the main thread for that Edit session, and completion and parameter hints look names up in it instead of evaluating. The main thread never evaluates bindings for them. This depends on that ticket's first step (built-ins become `Send`), so the evaluated context can cross threads.
2. **Cmd-/ before the context arrives** shows "Preparing completions…" and does nothing; the user presses again. A press is never remembered and applied later.
3. **Staleness.** A source reload queues a fresh prefetch. Slider moves do not: completion names can't change, and a parameter hint's named default that depends on a slider may show the value from when the context was fetched. That is accepted and left as a TODO in the code.
4. **One context.** The main thread keeps only the context of the program in Edit mode and drops it on leaving Edit mode. Contexts for every program were rejected: they would hold one prefix of the file per program, so memory grows quadratically with file size.

Derived from these, not separately decided:
- A prefetch that fails (an earlier binding doesn't evaluate) stores the error, and Cmd-/ shows "Can't complete: …" as now.
- A reply for an Edit session that has ended (left Edit mode, or entered it again) is dropped, like the stale results in [Where program processing runs](06-evaluation.md).
- ShowType and the per-keystroke parse stay on the main thread, per [Measure the Edit-mode helpers](09-measure-edit-helpers.md).
