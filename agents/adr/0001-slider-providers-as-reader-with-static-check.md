---
status: accepted
---

# Slider providers are a reader environment, guaranteed by a static check

A slider's provider is decided during evaluation, by a reader environment carried in the evaluator's dynamic scope: entering code written in a listing binding's text sets that binding as the provider of each name it lists, and a `slider` call reads its provider there, falling back to the program being played. A static analysis, run once the type checker accepts a program, guarantees in advance everything the old after-the-fact resolution checked at run time, so neither `slider` nor the fallback's substitution can fail. We chose this because resolving providers afterwards from recorded call paths was the most intricate part of the slider code, could fail at a keys instrument's note-on during a performance, and checked only the branches and notes one evaluation happened to reach.

## Decisions

- **Sound over permissive.** The analysis rejects every program that could break a rule below, even when that rejects programs that would have played correctly.
- **A static label.** A `slider` label is a string literal written in the call.
- **One listing per call.** At most one binding a call is reached through may list its label. Two listings are an error, not shadowing: an inner listing silently overriding an outer one was considered and rejected.
- **Specs fixed where their provider is.** A spec may depend on literals, top-level constants, and the arguments of calls made within its provider, as long as every call of that slider in the provider agrees. It may not depend on a keys instrument's key or velocity, or on anything else that varies between voices of one provider.
- **Listing bindings are checked from their definition alone.** A listing binding's specs may not depend on its own parameters, and it may not apply a function its callers give while it provides a slider. Editing one program therefore never makes another program fail.
- **The analysis lists the sliders.** It reports the sliders a program reaches, including all notes of a keys instrument, so the slider panel is fixed when the program loads.

## Considered Options

- **A second, abstract interpreter (the prototype in `d3fc817`).** Rejected because it reimplemented built-ins by hand and cut off at a fixed depth, which made it unsound. The chosen analysis is also an abstract interpreter, but it is exact on constants, applies the real built-in to known arguments, and needs no depth limit because the language has no recursion. It applies only `slider`, `map`, `reduce`, `unfold` and `share` itself, plus the identity `debug(label)` returns (so checking prints nothing). The only other knowledge of built-ins it needs is which ones can hand back a function they are given, `nth` and `concat`: those with polymorphic signatures, which a test keeps in step.
- **Running the real evaluator with symbolic values, or tracking binding time in the type checker.** Rejected because neither stays in one self-contained module.
- **Exactly one provider per call site in a program.** Rejected because it forbids a shared helper with a slider (such as a percussion envelope) inside two instruments in one kit, and the static label rules out passing the label in.
- **Specs at a listing binding taken from its callers** (one slider for each call of the binding, not one per binding). Deferred: this changes what "one slider" means, and is not something the analysis should allow.

## Consequences

- Hypothesis (unverified): a spec that would vary per voice can always be rewritten as a static spec plus arithmetic (for example `freq_hz * slider("ratio", 4, 1, 16, "exp")`), with only the number shown on the panel changing.
- Because one evaluation of a top-level binding is reused by many programs, the fallback provider is a placeholder, replaced by the program's slot when the waveform becomes that program's.
- The analysis counts a slider in a branch of `alt` that a constant trigger drops, even though evaluation never reaches it.
