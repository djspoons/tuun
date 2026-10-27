# Proposal 2: Give status messages a severity

**Status:** Implemented in `df3a7d2` ("Give status messages a severity and echo every post") and `e4edda8` ("Echo each program's errors from a load"), except for the slider-mode part noted under "Dropped". Both were rebased onto the background-processing rewrite (see "After rebasing" below).

## Problem

`AppState::message` was a single string, and the last write won:

- **Priority depended on effect order.** For an error to survive a later confirmation from the same event, the reducer had to post the confirmation *first* and rely on the error landing later. Some effects said so in comments ("Posted first so that an evaluation error … stays on screen"). That was an unwritten rule, built into the order of a `Vec<Effect>`, and easy to break.
- **The terminal echo was lossy.** The renderer printed the message only when it differed from the previous frame's. So a message overwritten within one event never reached the terminal, and repeating the same message ("Nothing to undo" twice) printed once. The per-program errors from the startup check were overwritten by "Errors in …" before the first frame, so their snippets were never printed.
- **The status line looked the same for an error as for "Metronome on",** and every message stayed until something replaced it.
- **Clearing happened three ways:** `ShowMessage(String::new())`, `state.message.clear()`, and plain overwriting.

## Proposal

Replace the string with `messages::Messages` (`src/lib/messages.rs`), where each message has a severity: Info, Warning or Error.

- **Batches.** `EffectRunner::dispatch`, `EffectRunner::run_all` and `EffectRunner::finish` each start a batch, so one user event, or one reply from the processing thread, is one batch. Within a batch, a post doesn't replace a more severe current message. A post in a later batch always replaces it. The ordering trick is no longer needed.
- **Echo.** Every post is queued. The main loop prints the queue each frame (`echo_messages` in `main.rs`), so each message reaches the terminal exactly once, with its full multi-line text. The renderer no longer prints.
- **Status line.** It shows the first line of the current message, coloured by severity:
  - Info: cyan (the existing inactive colour). Fades after `INFO_LIFETIME`, 5 seconds.
  - Warning: amber (new `WARNING_COLOR`). Stays until replaced or cleared.
  - Error: yellow (the editor's existing error colour). Stays until replaced or cleared.
- **Effects.** `Effect::ShowMessage(Severity, String)` carries the severity. A new `Effect::ClearMessage` replaces posting an empty string. `Messages::post_diagnostics` posts a list of diagnostics as an Error if any of them is an error, and as a Warning otherwise.

### Which severity

- **Error:** something failed: evaluation, saving, reloading, applying a keys instrument, and the startup "Errors in …".
- **Warning:** the user needs to act: type warnings, "File changed on disk", "Keys program hasn't evaluated", and a reload that came back with a warning.
- **Info:** everything else, including refusals like "Nothing to undo". These should fade rather than linger.

## After rebasing

`main` moved evaluation to a processing thread, so a result now arrives in a later batch than the event that asked for it. `main` already kept confirmations from hiding errors by deferring them to `on_success` effects. The rebase changed these things:

- **A reply is one batch.** `finish` posts the result's warnings or errors and runs `on_success`/`on_failure` in the same batch. Before, `main` set the warnings and then let `on_success` overwrite them (e.g. with "Playing A:1"), so they reached neither the status line nor the terminal. The test `a_warning_outranks_the_success_message_until_the_next_batch` covers this.
- **Held messages.** `Messages::hold` shows a message without echoing it, and a held message doesn't fade. `Messages::restore` puts back a replaced message without echoing it again, and any post replaces the restored message, even in the same batch. "Processing…" uses both: it is held while a job is pending, and the message it replaced comes back when the reply arrives. `Effect::ShowProcessing` holds it for refusals during processing, which would otherwise echo "Processing…" on every refused keystroke. Prefix prompts ("Annotation? …", "Loop A:1? …") are also held, so they don't fade while the app waits for the next key.
- **Load errors.** The startup sweep moved into `processing::load`, which kept only the failed programs' indexes. Their errors never reached the terminal, contrary to the claim above. `Loaded::failed` now carries each program's diagnostics. Each failure is echoed, prefixed with the program's name, before the "Errors in …" or "Reloaded … — failed: …" summary.
- **New messages from `main`.** "Not saved: … changed on disk (reload first)" and the other stale-file refusals are Warnings. Selecting a slot whose selection message is empty clears the message rather than posting "". Everything else new is Info.

### Still open

- **Loop refusals lose their severity.** `loop_refusal` returns a plain `String` for three reasons: an empty slot, a stale file and a keys program. All are posted as Info, so the stale-file reason fades instead of staying as a Warning, as the other stale-file refusals do. Fixing it means returning the severity with the reason.
- **Restored Info messages may have expired.** A restored message keeps its original post time. An Info message that was replaced by "Processing…" and restored more than 5 s later doesn't reappear. That seems right, but it's a change: `main` restored the text unconditionally.

## Dropped

The proposal also said: in Move Sliders mode, keep a warning or error visible one line above the slider readout. Run in the real app, this never showed anything. Every mouse move in Move Sliders mode posts an Info slider message (e.g. `X(0) = 0.467`). That's a new event, so under the batch rule it correctly replaces the error. Move Sliders mode keeps its old behaviour: the readout takes the status line.

Keeping errors visible there would need one of these. Both are trade-offs for the maintainer to decide:

- Errors stay current until navigation clears them, even across events. Then "Playing B:2" can't replace an error from A:1.
- Slider moves don't post messages while the readout is on screen. Then the terminal loses the slider values, and a Launchkey encoder turn in another editor mode still needs one.

## Behaviour changes

- In Move Sliders mode the terminal prints each slider message. Before, it printed the combined readout whenever it changed. The volume is about the same.
- Messages that used to stay on screen indefinitely now fade, e.g. a type signature from show-type. The full text is still in the terminal.

## Verification

Unit tests in `messages.rs` cover the outranking rule, later batches, clearing, expiry, holding, restoring and `post_diagnostics`. An effects test applies a reply with a warning whose `on_success` posts a confirmation, and checks that the warning stays current until the next batch. It fails if `finish` runs `on_success` as a batch of its own. The startup-load test checks that a failed program's error is echoed ahead of the summary.

The GUI was checked under Xvfb with simulated input before the rebase: the startup error shows in yellow, a selected program's name shows in cyan and is gone after 5 s, and Move Sliders mode works as before. It hasn't been rechecked since the rebase.
