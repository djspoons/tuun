# Sliders shared across programs

A slider should behave like one value plugged into several instruments at once:
move it, and every voice that uses it follows. Today a slider reaches only the
one program whose annotation declares it.

Vocabulary in this note is the one fixed by [CONTEXT.md](CONTEXT.md) —
*binding*, *program*, *voice*, *mark*, *slider*.

## The problem

A slider's identity is currently the pair `(WaveformSelector, label)`. The
selector does all the disambiguating work: `MarkId::Slider(String)`
(`ids.rs:71-80`) carries a bare label, and `apply_slider` emits
`Effect::UpdateSlider { selector: ProgramVoices(i), .. }`
(`actions.rs:1573-1577`), so a move reaches that one program's voice family and
nothing else. Both the coalescing map and the ramp baselines in
`flush_slider_updates` are keyed the same way (`effects.rs:41-60`).

That makes two same-labelled sliders on different programs distinct — which
`test7.tuun` relies on, declaring `cents`, `freq` and `D` twice each with
different values — but it also means a slider cannot be shared.

Sharing is expressed by declaring the slider on a *named* binding and having
other programs reference that binding, as in `test8.tuun`:

```
#{level_db=0, sliders=["mix:0.73046875:0:1"]}
m = mix;

#{level_db=0,keys}
_ = pm.epiano_keys_old(mix=m);
#{level_db=0,keys}
_ = pm.epiano_keys(mix=m);
```

`m` evaluates to `Marked(Slider("mix"), Const(0.73))`, so both keys programs
embed that mark. Two things then go wrong.

**The file does not evaluate at all.** `evaluation_bindings` keeps preceding
*named* bindings in scope and filters out only anonymous `_` ones
(`programs.rs:745-750`), then appends slider bindings for the target program's
own sliders only (`programs.rs:751-756`). So when A:2 is evaluated, `m = mix;`
is in its binding list with `mix` free. `build_context` evaluates each
`Binding::Definition` eagerly — `substitute` then `evaluate_closed`
(`eval.rs:567-573`) — and an unbound name in a definition body fails right
there, as `test_unbound_variable_fails_at_definition` pins (`eval.rs:825-842`).
A:2 and A:3 fail with `Variable 'mix' not found in context`.

*(Verified by reading each of those sites; the composition has not been
executed.)*

**Even if it evaluated, the value could not change.** Both keys programs would
hold a mark named `slider("mix")`, but the `Modify` goes only to
`ProgramVoices(A:1)`.

The existing TODO at `programs.rs:738-744` names this exact question and defers
it: *"there's an interesting question about what sliders in other bindings
mean… I think the right answer is that we should bind sliders uniquely in each
binding."* This note answers it.

## The model

A slider is declared on a binding and owned by it. The name it binds is
visible **only** inside that binding's own right-hand side; no other program
can write it. What other programs see is the binding's *value*, which carries
the mark. So a program opts into a slider by referencing the binding, never by
naming the slider.

The mark therefore has to identify the declaring binding, not just the label.

## Decisions

| Decision | Choice |
| --- | --- |
| Slider variable scope | Local to the declaring binding's RHS. Later programs reference the binding, never the slider name. |
| Mark identity | `MarkId::Slider { program: usize, label: String }` |
| Dispatch | One `Modify` on `AllVoices`; the `AllKeys` special case goes away |
| Value ownership | The declaring binding — i.e. today's `Program.sliders`, unmoved |
| Persistence | The declaring binding's annotation |
| Scoping mechanism | Wrap preceding slider-bearing bindings' RHS; leave own sliders as appended context |
| Slot occupancy | Unchanged: any annotated binding is a program |
| Encoder row | The active program's own sliders only |
| Play-time values | Loop `substitute` over every program's sliders |
| Ramp baseline | One per mark; re-seeded from the new set on reload; no note-on reseeding |
| Web paths | Unchanged |

### Mark identity

`MarkId::Slider { program: usize, label: String }`. The qualifier is the
program slot index rather than the index into `ProgramSet::bindings`, because
every slider-declaring binding is already a program: `display_name(program)`
renders the mark for free and satisfies the program-identifier rule in
`CLAUDE.md`, `Program.sliders` is already keyed this way so no new registry is
needed, and it shares an index space with `WaveformSelector::ProgramVoices`.

Note the two index spaces are not interchangeable — padding slots and
`skip_slots` mean binding N is not slot N.

### Dispatch

Once mark ids are unique the selector carries no information, so every slider
`Modify` goes to `WaveformSelector::AllVoices` (`ids.rs:48`) and
`waveform::substitute` is a no-op on voices that lack the mark. Both the
pending map and `last_slider_values` (`effects.rs:41-60`) collapse from
`(WaveformSelector, String)` to the mark id alone.

This subsumes the conditional second effect in `apply_slider`
(`actions.rs:1581-1589`), which mirrors a slider onto `AllKeys` when the
installed keys instrument came from that same program. It exists only because
`ProgramVoices(i)` cannot reach key voices; with a unique mark it is redundant.

### Scoping mechanism

There is no `Expr::Let`: `let p = e in body` is surface sugar the parser
desugars into an immediately-applied one-parameter lambda, which the printer
re-sugars on the way out (`as_let_binding` / `fmt_as_let`, `expr.rs:804-838`).
"Wrapping" a binding's RHS therefore means building that application — no
parser or AST change, and it prints back as readable `let` if displayed.

Two positions, two mechanisms, both inside `evaluation_bindings`:

- **A preceding binding that declares sliders** has its RHS wrapped, so the
  slider name is in scope for that definition alone.
- **The target program's own sliders** stay as appended context entries
  (`append_slider_bindings`, `programs.rs:751`), which is already correct.

Keeping the second mechanism rather than wrapping uniformly is a code-size
call: `evaluation_bindings` is the single choke point that all five consumers
in `environment.rs` already call (lines 402, 505, 560, 590, 684). Moving
own-slider binding out to the expression would mean wrapping at each of those
five sites plus `web_checker.rs:320`, for a helper of the same size.

Because `append_slider_bindings` appends *after* every preceding binding, and
`build_context` evaluates in list order, a program's own sliders are never in
scope while its predecessors are evaluated. There is no accidental capture: a
program declaring its own `mix` does not bind a preceding binding's free `mix`.

The `retain` filter on `_` bindings (`programs.rs:745-750`) stays, but its
justification changes. Its stated reason — dodging the slider question — is
resolved here. It is kept because `_` bindings are unreferenceable, evaluating
every preceding program on every evaluation is wasted work, and one broken `_`
program should not poison later ones. The TODO above it is rewritten, not
deleted.

### Play-time values

`substitute_current_slider_values` takes the program's own `&ProgramSliders`
(`player.rs:37-52`), which in `test8.tuun` is empty for A:2 and A:3 — so `mix`
would never be refreshed and a voice would start at whatever was current when
A:2 was last evaluated. It needs the whole `ProgramSet`.

It does this by looping `waveform::substitute` over every program's sliders. A
mark that is not present costs one fruitless walk and changes nothing. The
alternative — collecting the marks actually present, then substituting only
those — would mean writing a sixth full-tree visitor alongside
`initialize_state`, `remove_state`, `set_state` and `substitute`, each a ~65
line match over every variant. The slider counts do not justify it.

This works because `substitute` replaces the *contents* of a `Marked` node and
keeps the wrapper (`waveform.rs:413-419`), so baking current values in at play
time leaves the voice modifiable, and because `optimizer::optimize` preserves
`Marked` nodes rather than folding them (`optimizer.rs:428-434`).

### Ramp baselines

The slider worker is spawned once and owns `last_slider_values` for the life of
the process; `ReloadSource` never touches it. With a program-qualified mark
that matters: after a reload, slot 3 may be a different binding entirely while
the worker still holds a baseline for `{3, "mix"}` from the old file.

`Effect::ReloadSource` therefore replaces the worker's baselines with those of
the set it just loaded, discarding every mark the new set does not declare. It
already validates before silencing, then stops and clears all voices
(`effects.rs:270-282`), so nothing is sounding at that moment and the new
set's own slider positions are the right thing to ramp from next.

Merely *clearing* would also be correct — with no voices left there is nothing
to ramp from — but it would leave a reload meaning something different from a
startup, which seeds each baseline from the slider's position. Re-seeding makes
the two the same, and costs one helper (`effects::slider_values`) that both
paths call. This finally gives `SliderEvent::SetInitialValues` a producer — it
is declared at `effects.rs:32` and handled in `main.rs`, but nothing has ever
sent it; it is renamed `SetBaselines`, which is what it does.

The note-on reseeding (`UpdateInitialValues` from `PlayNoteOn` / `PlayNoteOff`,
`effects.rs:352-396`) is retired: one shared baseline per mark makes it
unnecessary, and it exists only to paper over the per-program keying being
removed.

### What does not change

**The encoder row.** The eight encoders map 1:1 onto the active program's own
sliders (`midi_input.rs:26-39`) and continue to. In `test8.tuun` you keep A:1
active so the encoders show `mix`, and install A:2 or A:3 as the keys
instrument from the Keys Installer pads — `apply_install_keys` emits only
`Effect::InstallKeys` and never touches `active_program_index`
(`actions.rs:976`). Showing a union of own plus transitively-referenced
sliders would need a reference closure, a re-addressing of
`Action::SetSliderNormalized { program, slider_index }`, and an answer for
overflowing eight encoders. Not now.

**Persistence.** Moving a slider owned by A:1 while A:2 is active writes A:1's
annotation, and `splice` already collects annotation edits for *every* program,
not just the save target — `non_active_program_divergence_persists_on_any_save`
pins exactly that (`programs.rs:1809-1837`). No new machinery.

**Stale marks.** A live voice can hold a mark whose slider was since renamed or
deleted, and a program-qualified mark is positional. Neither matters:
annotations can only be changed by editing the file, which reaches the app as a
reload, and `ReloadSource` stops all voices before swapping the set
(`effects.rs:277-279`). Program positions do not shift under `splice`.

**The web paths.** A web component hosts one expression with one slider set, so
sharing cannot arise. The wasm runtime builds its marked node inline
(`wasm.rs:227-233`) and never calls `append_slider_bindings`, so it is
untouched; `web_checker.rs:320` passes a fixed dummy qualifier.

## Incidental fixes

These are wrong today, independently of this feature.

- **`environment.rs:693-694`** claims the expressions passed to
  `apply_note_function` are *"closed except for references to sliders, which
  are bound at their current values"*. They are fully closed.
  `evaluate_program` substitutes the whole context into the program expression
  before storing it (`environment.rs:590`, `604`, `637`), and `substitute`
  descends into lambda bodies, shadowing parameters first (`eval.rs:76-83`). So
  a slider used outside the lambda (`let ratio = D/2 in fn(k,v) => …`) and one
  used inside its body are both already replaced by a `Marked` node at
  evaluation time.
- **`environment.rs:708`**, the `append_slider_bindings` call inside
  `apply_note_function`, is dead for the same reason: it binds names that are
  no longer free, on both call paths. It goes, along with the now-unused
  `sliders: &ProgramSliders` parameter.
- **`programs.rs:738-744`**, the TODO this note answers, is rewritten to
  document why `_` bindings are still filtered.

## Verification

Build gate per `CLAUDE.md`: `cargo build`, `cargo build --benches`,
`cargo test`, `cargo fmt`, `cargo clippy`.

Behaviour:

1. `test8.tuun` works — one `mix` slider driving both keys instruments, with
   the value carrying across a switch between them.
2. The existing suite stays green, `test7.tuun` included.
3. New tests for the three invariants:
   - a slider variable is invisible outside its declaring binding;
   - two same-labelled sliders on different programs stay distinct marks
     (`test7.tuun`-shaped);
   - a slider move reaches a voice belonging to a different program.
4. A regression test that a named binding carrying `sliders=` no longer makes
   later programs fail with `Variable 'mix' not found in context` — the bug
   this note exists to fix. Without it, a future refactor restoring the
   `retain` hack would pass silently.

`evaluation_bindings_filters_anonymous_and_appends_sliders`
(`programs.rs:1555`) pins today's behaviour and needs updating; it is the
natural home for (3) and (4).

## Parked

- **Should a binding that exists only to declare sliders be playable?** It is
  visible and occupies a slot either way. `m = mix;` evaluates to a bare
  `Const`, so playing it emits DC. Nothing here depends on the answer.
- **Can annotations be edited without a full reload?** Today they cannot, which
  is what makes stale marks a non-issue above. If in-app annotation editing
  ever lands — the deferred wide-mode work would be the vehicle — the stale
  mark and reload-baseline questions both reopen.
