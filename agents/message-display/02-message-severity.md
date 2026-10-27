# Proposal 2: Give status messages a severity

**Status:** Implemented in `cf17c26` ("Give status messages a severity and echo every post"), except for the slider-mode part noted under "Dropped".

## Problem

`AppState::message` was a single string, and the last write won:

- **Priority depended on effect order.** For an error to survive a later confirmation from the same event, the reducer had to post the confirmation *first* and rely on the error landing later. Some effects said so in comments ("Posted first so that an evaluation error … stays on screen"). That was an unwritten rule, built into the order of a `Vec<Effect>`, and easy to break.
- **The terminal echo was lossy.** The renderer printed the message only when it differed from the previous frame's. So a message overwritten within one event never reached the terminal, and repeating the same message ("Nothing to undo" twice) printed once. The per-program errors from the startup check were overwritten by "Errors in …" before the first frame, so their snippets were never printed.
- **The status line looked the same for an error as for "Metronome on",** and every message stayed until something replaced it.
- **Clearing happened three ways:** `ShowMessage(String::new())`, `state.message.clear()`, and plain overwriting.

## Proposal

Replace the string with `messages::Messages` (`src/lib/messages.rs`), where each message has a severity: Info, Warning or Error.

- **Batches.** `EffectRunner::dispatch` and `EffectRunner::run_all` each start a batch, so one user event is one batch. Within a batch, a post doesn't replace a more severe current message. A post in a later batch always replaces it. The ordering trick is no longer needed.
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

## Dropped

The proposal also said: in Move Sliders mode, keep a warning or error visible one line above the slider readout. Run in the real app, this never showed anything. Every mouse move in Move Sliders mode posts an Info slider message (e.g. `X(0) = 0.467`). That's a new event, so under the batch rule it correctly replaces the error. Move Sliders mode keeps its old behaviour: the readout takes the status line.

Keeping errors visible there would need one of these. Both are trade-offs for the maintainer to decide:

- Errors stay current until navigation clears them, even across events. Then "Playing B:2" can't replace an error from A:1.
- Slider moves don't post messages while the readout is on screen. Then the terminal loses the slider values, and a Launchkey encoder turn in another editor mode still needs one.

## Behaviour changes

- In Move Sliders mode the terminal prints each slider message. Before, it printed the combined readout whenever it changed. The volume is about the same.
- Messages that used to stay on screen indefinitely now fade, e.g. a type signature from show-type. The full text is still in the terminal.

## Verification

Unit tests in `messages.rs` cover the outranking rule, later batches, clearing, expiry and `post_diagnostics`. An effects test runs a failing evaluation and a confirmation in one `run_all`, and checks that the error stays current until the next run.

The GUI was checked under Xvfb with simulated input: the startup error shows in yellow, a selected program's name shows in cyan and is gone after 5 s, and Move Sliders mode works as before.
