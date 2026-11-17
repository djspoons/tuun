# Slider builtin

A design for replacing the per-program slider bindings with a `slider`
builtin. Agreed in a design interview; see "Deferred" for what is still open.
Glossary terms (**Slider**, **Binding**, **Program**, **Annotation**, …) are as
defined in [`agents/CONTEXT.md`](../CONTEXT.md).

## Motivation

Designing instruments means experimenting with parameters deep inside library
functions. Today a slider is declared in a binding's annotation and bound as a
variable in that binding's scope only, so reaching a parameter inside an
instrument means threading it through every call between the annotated
binding and the place it's used. A builtin that can be called anywhere removes
that plumbing.

## How sliders work today

For reference when implementing (as of commit `1dae9b2`):

- An annotation `#{sliders=["Q:0.707:0.1:1.1"]}` declares a slider, either
  linear (`label:init:min:max`) or with a Tuun mapping function
  (`label:normalized-init:fn(x) => …`). Parsed in `src/lib/parser.rs`
  (`parse_sliders_internal`).
- Each slider becomes a binding `Q = Marked(Slider { program, "Q" }, Const(v))`
  in scope only inside its declaring binding (`ProgramSet::own_slider_bindings`
  and `evaluation_binding` in `src/lib/programs.rs`).
- A slider's identity is `MarkId::Slider { program, label }`
  (`src/lib/ids.rs`).
- Current values are substituted into the marks at play time and at keys
  note-on/off (`player::substitute_current_slider_values`); moves become ramp
  `Modify` commands (`slider::make_ramp`, `effects::flush_slider_updates`). The
  optimizer never moves code out of a `Marked` node.
- Slider state is a list parallel to the annotation, per program
  (`ProgramSliders` in `src/lib/programs.rs`). Encoders and the mouse X/Y
  fallback map by position in that list.
- Changed slider values are written back into the annotation whenever an
  action saves the source (`Effect::UpdateSource` → `ProgramSet::splice` →
  `annotation_edits` / `sliders_edit`).
- The web component identifies sliders by a global string name
  (`MarkId::Slider(String)` in `src/lib/wasm.rs`) and configures them with a
  `sliders='[…]'` attribute.
- Custom mappings in the repo's examples are all of the form `a * pow(b, x)`.

## Design

### The builtin

```tuun
slider("mix", 0.5)                            // linear over 0–1
slider("Q", 0.707, 0.1, 1.1)                  // linear over min–max
slider("cutoff", 800, 100, 11000, "exp")      // curve: "linear" or "exp"
```

- One builtin, three arities (2, 4, 5). Builtin signatures already support
  conjuncts of different arities (the built-in `-` has unary and binary forms;
  `src/lib/signatures.rs`, selection in `src/lib/infer.rs`
  `select_conjunct`), and builtins dispatch on `arguments.len()`.
- No named arguments.
- The label must be a string and the default, min and max must be scalars,
  using the numeric sorts. They need not be literals: any expression that
  evaluates to a scalar is accepted, so helpers like `fn(name) => slider(name, …)`
  work.
- These are evaluation errors at the call: a default outside `[min, max]`;
  `min >= max`; `"exp"` with `min <= 0`; a curve other than `"linear"` or
  `"exp"`.
- Every call that reaches the same slider (same owner, same label; see
  below) must agree on default, range and curve. A disagreement is an
  evaluation error naming both values. Calls that agree share the slider on
  purpose.
- The result is a waveform: a marked placeholder holding the value, with the
  default as the value a check or a dry evaluation sees. Evaluation stays
  pure; the current value is substituted when the sound plays, and moves reach
  the playing sound as ramps, as today. As today, a slider can't be used where
  evaluation needs a plain number (list sizes, `on_beats` counts, branching on
  its value).

### Ownership

Recorded as the **Slider** entry in [`agents/CONTEXT.md`](../CONTEXT.md):

- Among the bindings a sound is built through, the one whose annotation lists
  the slider's label owns it. If none does, the program being played owns it.
- Every voice built through the same owner shares one view of the slider;
  voices built through different owners never share one.
- If two bindings a sound is built through both list the same label, the
  program that reaches both fails to evaluate, with a diagnostic pointing at
  both annotations. Programs that reach only one of them are unaffected.

Example: `pad` is a library function calling `slider("cutoff", …)`.
`A:1` is `delayed_pad = fn(freq) => … pad(…)`, `A:2` is `_ = delayed_pad(220)`,
and `A:3` is `_ = delayed_pad(330)`.

| Annotations listing `cutoff` | Result                                       |
|------------------------------|----------------------------------------------|
| none                         | `A:2` and `A:3` each own their own `cutoff`  |
| `A:1`                        | `A:2` and `A:3` share `A:1`'s `cutoff`       |
| `A:2` only                   | `A:2` owns one; `A:3` owns its own           |
| `A:1` and `A:2`              | `A:2` is an error; `A:3` shares `A:1`'s      |

This changes one behavior: today a program that uses another program's
binding shares that binding's sliders without listing anything. Under this
design it shares them only if the binding's annotation lists them, which it
will, since that annotation is where they were declared before.

Why this rule: a library function has no program slot, so a slider inside one
can't be owned by "the binding that contains the call" the way annotation
sliders are today. Owning by the played program gives each program an
independent value by default, and an annotation on a shared binding opts its
users into sharing. Two listings would leave the owner ambiguous, so it's an
error rather than an override.

Hypothesis (unverified): the slider's mark can be produced with only its label,
and qualified with its owner when the program's waveform is played, the same
moment current values are substituted today. Evaluation and cached shared
waveforms then never depend on which program reads them.

### The annotation

```tuun
#{sliders=["cutoff:1200", "Q"]}
```

The annotation no longer creates bindings. It does two jobs: it sets the
encoder order, and it records saved values.

- An entry is a label with an optional saved value in real units. A bare label
  means "use the code's default".
- A saved value outside the code's range is clamped, with a warning.
- An entry that the program's sound never reaches gets a warning, but keeps its
  encoder position and its saved value.
- The web component's `sliders='[…]'` attribute uses the same entry format.

### Encoders and display

For the selected program, encoders (and the mouse X/Y fallback, which takes the
first two) are assigned in this order:

1. Sliders listed in the program's own annotation, in annotation order.
2. Sliders the program owns but doesn't list ("auto-shown"), in the order
   their calls first appear.

Values are displayed in real units (Hz, Q), not 0–1. Display names follow the
usual program-identifier convention in `CLAUDE.md`.

A slider owned by an earlier binding (`A:1` above) is adjusted by selecting
that binding's slot. Whether it should also appear on `A:2`'s encoders is
deferred; see below.

### Saving

- Same trigger as today: whenever an action saves the source, slider values
  that differ from their owner's annotation are written into it.
- An auto-shown slider is written only once it has been moved; until then it
  stays at the code's default and nothing is added to the file.
- If ownership later shifts so that two bindings list the same label (e.g. an
  auto-saved `"cutoff:1200"` on `A:2`, then `"cutoff"` added to `A:1`), the
  ownership error names both annotations and the user deletes one entry. No
  special handling.

### Migration

A hard break, with no compatibility mode:

- An old-style entry (`label:init:min:max` or one with a `fn(x) => …`
  mapping) is a parse error whose message says what to write instead.
- The same change converts every in-repo use: `lib/v0/demos/sliders.tuun` and
  other `.tuun` files, the `docs/` pages and their web components (checked by
  `check-web-examples.sh`), and the README.
- `slider::denormalize`'s Tuun-function evaluator goes away; curves are a
  closed set.

### Implementation constraint

Slider state should move out of each `Program` (`ProgramSliders`) into one map
in `ProgramSet`, keyed by owner and label. Hypothesis (unverified): this keeps
the encoder ordering, auto-showing, and the deferred cross-owner display
simple, since each becomes a filter or ordering over the program's reachable
sliders rather than a separate structure.

## Deferred

- **Showing sliders owned by earlier bindings on a program's encoders.** With
  `cutoff` owned by `A:1`, should selecting `A:2` show it (after `A:2`'s own,
  labeled with its owner) so it's reachable from a slot that makes sound? The
  concern: turning that encoder while `A:2` is selected also changes `A:3`.
  Decide at a later implementation stage; the first stage shows only sliders
  the selected program owns.
- **Removing named arguments from the language**, if the builtin turns out to
  cover the parameter-plumbing case they were meant for. Out of scope here.
