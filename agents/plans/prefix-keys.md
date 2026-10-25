# Plan: prefix keys for quitting and annotations

Add Emacs-style **prefixed commands** (see `agents/CONTEXT.md`: *prefix key*, *prefixed command*) to the SDL desktop app. The web build is out of scope.

Goals:

- Quitting takes two presses, Cmd-Q Cmd-Q, so a performance isn't stopped by accident. Ctrl-C no longer quits.
- Cmd-N k / u / d change the active program's annotation without opening another editor.

## Behavior

### Prefix keys in general

- Pressing a prefix key shows its prompt on the message line and waits. There is no timeout.
- The next key either completes a prefixed command or cancels the prefix. A key that cancels is swallowed: it does nothing else, including Escape, and the prompt is cleared. Lone modifier presses (Cmd, Shift, …) neither complete nor cancel.
- While a prefix is pending, the next key is never read as a piano note in Keys mode (Cmd-N d must not play D#4).

### Quit: Cmd-Q Cmd-Q

- On macOS, SDL's app menu turns Cmd-Q into `Event::Quit`. The window's close button also sends `Event::Quit`, and per SDL's documentation (unverified in Tuun) so does SIGINT. **Every `Event::Quit` counts as a press of the quit prefix key.** The first press shows `Quit?`; a second `Event::Quit` while `Quit?` is pending exits. As a result, the close button and Ctrl-C in the launching terminal also need two presses. This is intended.
- Quitting always takes two presses, whether or not anything is playing, in every editor mode.
- Exiting still saves the active program first (`Action::Exit` already emits `Effect::UpdateSource` before `Effect::Exit`). Keep that.
- Remove both Ctrl-C → `Action::Exit` bindings in `src/lib/sdl2_input.rs`: the Keys-mode one and the all-modes one.

### Annotation: Cmd-N

- Available in the Select and Keys editor modes. In every other editor mode, Cmd-N is refused silently, the same way `refused_by_ownership` refuses commands on an owned program, and no prompt appears.
- Prompt: `Annotation? (k)eys (u)p (d)own`.
- All three act on the active program. On an empty slot (one with no binding), do nothing and show a message.
- If the backing file changed on disk since the last read or write, refuse with the same message `splice` gives. The user presses Cmd-R first.
- None of the three is undoable; each has an inverse.

#### Cmd-N k: toggle `keys`

- Adds `keys` to the active program's annotation, or removes it. This changes only that program's kind.
- **No reload.** Handle it like `level_db`: change the kind on the in-memory `Program`, write the annotation back to the file through `splice`, then re-evaluate just that program (`Effect::EvaluateProgram`), since how it evaluates depends on its kind.
- It doesn't install or uninstall the keys instrument; `K` still does that. One exception: if the program is the **installed keys instrument** and loses `keys`, uninstall it exactly as `K` does (`apply_install_keys`, `src/lib/actions.rs`), which stops held keys with a stop ramp.
- Voices of a program that gains `keys` keep sounding; they are already-evaluated waveforms.

#### Cmd-N u / d: change `skip_slots`

- `u` lowers the active program's `skip_slots` by one, moving it up to a lower slot. `d` raises it by one.
- Only the active program's annotation changes, so **every later program moves with it**. The main use: after adding a program in another editor and pressing Cmd-R, shift the following programs back up.
- `u` at `skip_slots=0` does nothing and shows a message. `d` is refused, with a message, if any program would land past the last slot (64).
- The selection follows the moved program, which may move it into another bank.
- Program indices shift, so do what Cmd-R does: write the file, then reload it through the same path as `Effect::ReloadSource`. That stops every voice and uninstalls the keys instrument.

## Implementation sketch

1. **State.** Add a pending prefix key to `AppState`, for example `pending_prefix: Option<PrefixKey>` with `PrefixKey::{Quit, Annotation}`. Showing the prompt sets `state.message`; completing or cancelling clears both.
2. **Classification (`src/lib/sdl2_input.rs`).**
   - `Event::Quit` → `Action::Exit` when `Quit` is pending, otherwise an action that sets the `Quit` prefix.
   - Before the existing per-mode match, and before the Keys-mode piano branch: if a prefix is pending, classify the key as the completing command or as a cancel. Skip modifier-only scancodes. Make sure a swallowed key also produces no `TextInput` effect: SDL sends a `TextInput` after the `KeyDown` for printable keys, so the second key of Cmd-N k must not insert or select anything. (Hypothesis, unverified: with Cmd held for the first key only, `k` arrives as both a `KeyDown` and a `TextInput "k"`; in Select mode the `TextInput` would enter Keys mode.)
   - Cmd-N (GUI modifier + `Scancode::N`) in Select or Keys mode → set the `Annotation` prefix.
3. **Actions (`src/lib/actions.rs`).** New actions, for example `ToggleKeysAnnotation(usize)` and `ChangeSkipSlots { program_index, delta }`, routed through `refused_by_ownership`.
4. **Programs (`src/lib/programs.rs`).**
   - `keys`: make `Program::kind` settable, and teach `annotation_edits` to add or remove the `keys` annotation, the way `level_edit` handles `level_db`. Removing the last annotation needs the same treatment `skip_slots_edit` gives it: a binding with no annotation no longer occupies a slot, so swap in `level_db=…`.
   - `skip_slots`: an edit that sets one binding's `skip_slots`. Reuse `skip_slots_edit`, then write the file and reload.
5. **Docs.** Update the key list in `README.md`: add Cmd-Q Cmd-Q and the Cmd-N commands, and remove any mention of Ctrl-C.
6. **Tests** (in `sdl2_input.rs` and `actions.rs`, following the existing style):
   - One `Event::Quit` doesn't exit, and a second one does; any other key in between cancels.
   - Ctrl-C no longer exits, in Select, Edit or Keys mode (update the existing Ctrl-C tests).
   - Cmd-N then `d` in Keys mode produces no `NoteOn`.
   - Cmd-N is refused in Edit and Record modes.
   - `k` adds and removes `keys`, re-evaluates, and uninstalls the installed keys instrument when it loses `keys`.
   - `u` at 0 and `d` past slot 64 are refused; `u` and `d` move the program and every later one, and the selection follows.

## Before starting

- On a Mac, confirm that Cmd-Q arrives as a single `Event::Quit` with no `KeyDown`. If a `KeyDown` for Q also arrives, it must not count as a second press.
