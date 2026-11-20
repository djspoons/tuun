# Review: slider providers and static slider analysis

A review of commits `4d9add1` and `38a4805` on `main`, taken together as one change. It looks for clarity in the code, least surprising behavior in the app, and code simplicity.

The design holds up, and it's much simpler than resolving owners afterwards from recorded call paths. A provider is now a reader environment carried in `eval::DynamicScope`, and the static analysis (`slider_analysis::analyze`) guarantees what the old resolution checked at run time. Moving the `share` counter out of a thread-local into the same scope is a nice consolidation too.

Measurements: across the 86 programs in `lib/v0` the check reports no slider findings. The slowest whole check, type checking included, took about 100 ms in a release build.

Each item below is open unless marked otherwise.

## Bug: an installed keys instrument loses its helper's slider after an edit elsewhere

**Addressed:** `Plays::KeysInstrument` carries a `programs::KeysInstrument`, the function with the listings from its `EvaluationInput`, and `apply_note_function` takes it in place of the current providers. Test: `keys_instrument_keeps_its_providers_after_an_earlier_slot_grows` in `src/lib/environment.rs`.

Verified with a probe test against `38a4805`.

`PlayNoteOn` (`src/lib/effects.rs`, around line 534) finds providers from the current set (`state.programs.slider_providers()`). The keys function, though, is a snapshot whose closures carry file offsets from when it was evaluated. When an edit earlier in the file is spliced back in, every later slot's text range moves, so the old offsets point into the wrong slot.

Steps that reproduce it:

1. A file whose A:2 is `h = fn(n) => $n * slider("q", 0.5)`, listing `q`, and whose A:3 is a keys instrument calling `h`.
2. Evaluate the keys function and apply it: the slider is `q of A:2 (h)`.
3. Make A:1's text longer, `splice(0)`, and apply the same function again: the slider is now `q of A:3`.

That slider has no state, so the note plays its default and h's encoder stops doing anything. Nothing is reported.

**Fix:** find providers with the slot ranges that were current when the function was evaluated. For example, `Plays::KeysInstrument` could carry the `Listings` from its `EvaluationInput`, and `PlayNoteOn` and `PrintEvaluatedProgram` would use those. Capturing them at install time isn't enough, because a splice can also happen between evaluation and install.

## UX: a listing on a function binding is a trap

The "list it on a shared helper" pattern that the provider errors steer users toward has three rough edges:

- **The slot shows as failed.** A slot like `#{sliders=["q"]} pad = fn(f) => …` fails with `expected numeric, found (waveform) -> waveform`. That's older behaviour, but the new design and its errors now push people toward these bindings.
- **An alias silently provides nothing.** `#{sliders=["cutoff"]} alias = lead;` gets no warning. A:1's encoder shows a dead "cutoff", and the program playing `alias(220)` auto-shows its own separate "cutoff".
- **The "lists slider … but nothing reaches it" warning rarely appears.** `Providers::warnings` (`src/lib/slider_providers.rs:111`) only warns about the played program's own entries, so it never fires for a helper slot. It doesn't show at load either, per the TODO in `processing::load`.

**Suggestion:** check a slot that lists sliders and evaluates to a function as a provider. Run `check_alone` as that slot's own check, and replace the "evaluates to a function" error with "lists `X`, but provides no slider `X`" where that applies. The analysis already knows exactly which labels a listing provides on its own, so this would also catch the alias case.

**Addressed:** a waveform slot that lists sliders may evaluate to a function: the type checker accepts a function there (`Expectation::PlayableOrFunction`), and evaluation returns `Plays::Provider`, which plays nothing and isn't shown as failed. The slider check runs on the slot's value alone (`slider_analysis::Role::Provides`). The slot's sliders are the ones it provides, so its encoders show them once it is evaluated. Each entry it lists but doesn't provide is an error, "lists slider `X`, but provides no slider `X`", which catches the alias case. Unreached entries of a slot that plays are still warnings (`Providers::unreached`). Those warnings still don't show at load, per the TODO in `processing::load`. Tests: `a_listing_function_provides_its_sliders_and_plays_nothing`, `a_listing_function_that_doesnt_provide_an_entry_is_an_error` and `a_function_that_lists_nothing_doesnt_play` in `tests/integration/slider.rs`.

## Simplifications

- **Two `Reached` types.** `slider_analysis::Reached` (with a `Provider`) and `slider_providers::Reached` (with a slot index) differ only by the `index` closure in `analyze_sliders`. If `slider_analysis::Program` took the played slot's index, the analysis could report slot indices directly, and one type and that mapping would go away. `Played` vs `Listing` is still needed in evaluation, as the placeholder for bindings that are evaluated once and shared, but not in what the analysis returns.
  **Addressed:** `slider_analysis::Program` takes the played program's id, and `slider_analysis::Reached` holds provider ids; `slider_providers::Reached` is gone. Keying specs by id also fixed a missed error: a slider the played program provides both unlisted (from a top-level binding) and through its own listing was two sliders to the analysis but one in evaluation (test: `a_slider_the_played_program_lists_is_the_one_it_provides_unlisted`).
- **Dead code.** `MarkId::provided_by` (`src/lib/ids.rs:117`) is never called; `slider_providers::provide` does the same job. The doc comment on `PlayedSlider` links to the dead one.
  **Addressed:** removed; `PlayedSlider`'s doc links to `slider_providers::provide`.
- **`apply_note_function` returns `Vec<Error>` but never more than one error.** That's left over from the runtime resolution. Going back to a single error removes the diagnostic-collecting code in both `effects.rs` callers.
  **Addressed:** it returns one `expr::Error`.
- **`DynamicScope::enter`** clones the providers list and allocates the labels on every closure entry into a listing's text, even when the listing already provides them. Return early when nothing changes. This is minor and only shows up with `map` over large lists.
  **Addressed:** `enter` returns the scope unchanged when the listing already provides all its labels, and `Providers::listings` builds each slot's labels once rather than on each lookup.

## Docs and claims

- **The ADR overstates one claim.** Its consequences say "Evaluation never fails on a slider." The `slider` built-in still fails on an invalid spec (`src/lib/builtins.rs`, "A program may be evaluated without the slider check"), so in warnings mode an invalid spec does fail evaluation. The claim needs a carve-out for invalid specs.
  **Addressed:** the consequence now says evaluation fails on a slider only when its spec is invalid.
- **The `eval.rs` `providers` test module has two stacked doc summaries.** "Tests of the running code that evaluation hands built-ins…" is left over.
  **Addressed:** removed.
- **The `slider` built-in's doc comment** has an unwrapped line over 100 characters.
  **Addressed:** wrapped.

## Tests

`src/lib/slider_analysis.rs` (about 1,700 lines) has no unit tests. Its coverage is the integration tests (`tests/integration/slider.rs`), the soundness fuzz (`slider_soundness_fuzz` in `tests/integration/checker_fuzz.rs`) and one doc example, all using the real built-ins. The repeat-until-stable logic (`iterate` / `covers`) and `join` / `merge` are the subtle parts. A few unit tests with stand-in built-ins would pin them down independently of the prelude, per CLAUDE.md.

**Addressed:** `slider_analysis` has unit tests of `covers`, `iterate` (through `reduce` and `unfold`), and `join`/`merge` (through `if` on a note), with stand-ins for the built-ins and operators the check knows by name. Each test was checked to fail when the logic it pins is broken.

## Minor

- **Sliders are never forgotten.** `ProgramSet::sliders` keeps every slider ever reached. After A:2 stops using `pad`, A:1's encoder still shows and moves a "cutoff" that nothing reads.
  **Addressed:** recording an outcome forgets each slider that no program's latest evaluation reaches (`ProgramSet::forget_unreached`). The exception is a slider whose providing slot has no evaluation, for example while it fails, which keeps its position. A moved position isn't lost: a splice saves it in the annotation before the edited program is evaluated again, and a slider reached again starts at its saved value. A forgotten slider's encoder does nothing, like a listed slider nothing has reached yet. With the listing-function change above, a slot's listed slider is usually reached by the slot's own evaluation, so this mostly applies to a slot that stops reaching an entry it lists.
- **False positive on known parts of an unknown list.** `slider("q", nth(0, [0.1, note / 1000]))` is reported as depending on the key a note plays. This is expected under "sound over permissive".
- **`debug` is recognised by its name** (`name.starts_with("debug(")` in `slider_analysis`). This is string coupling to how `builtins::debug` names its result.
  **Addressed:** `builtins::is_debug_result` sits beside the function that names `debug`'s result, and the analysis calls it.

## Suggested order

1. The keys note-on bug.
2. Listing slots that evaluate to functions, since it most affects what users see.
3. The simplifications and doc fixes, which are small and independent.
4. Unit tests for `slider_analysis`.
