# Proposal 3: Show the results of controller actions on the Launchkey display

**Status:** Not implemented. Depends on proposal 2 (message severity), which is in place.

## Problem

The Launchkey display shows exactly two things (`Launchkey::set_daw_mode_display`, `Launchkey::set_encoder_display`):

- The DAW pad sub-mode label ("Clip Launcher", "Keys Installer", "Sequencer") when the performer enters or cycles DAW pad mode.
- Each encoder's name and value. In Plugin mode that's the slider label and value. In Mixer mode it's `level` and the dB value.

Everything else the controller shows is through LEDs: repeat, launch mode, record and metronome (`midi_input::update_launchkey_state`).

So when the performer presses a pad or button and something goes wrong, the controller gives no sign. They have to look at the laptop. Examples:

- A top pad press on a program with an evaluation error, or one that isn't sequenceable.
- Record when the file changed on disk.
- The result of a take: "Recorded 3 notes into A:2", "Discarded take", "Nothing recorded".
- Undo or redo from the Undo button: "Nothing to undo", "Undid change to kick".
- Sequencer page changes from the pad page buttons: "Sequencer beats 5–8".
- Entering the Sequencer DAW pad sub-mode on a program that isn't `[steps] | on_beats(w)`. The status line explains why; the display only says "Sequencer".

These are all results of controller gestures, but they only reach the GUI.

Separately, all eight Mixer-mode encoders are labelled `level`, while the status line says `level(kick) = -6 dB`.

## Proposal

1. **Send messages from controller actions to the display.** When a batch of actions came from the Launchkey (it was dispatched from `midi_input::classify`), and the batch's current message is worth showing, write its first line to the display. Reuse the DAW-mode display target, or a temporary-display arrangement if the device has one.
   - Errors and warnings always go to the display.
   - Info goes only for results the performer can't see otherwise: take results, undo/redo, page changes. Things the LEDs or the sound already show, like "Playing A:2", don't need it.
2. **Name Mixer-mode encoders after their program.** Use `programs.display_name(i)` (e.g. `kick` or `B:3`) as the encoder name in place of `level`.
3. **Standardize message wording** so it fits a short display and reads the same everywhere: a verb plus a name, e.g. "Playing kick", "Stopped kick", "Error in kick". Today the nouns vary ("Playing waveform A:2", "Stopped program A:2", "Removed pending waveform for program A:2", "Recorded 1 note into A:2"). Slider messages also use a 0-based slider index (`cutoff(0) = 440`), while level messages use the program name.

## Design notes

- Proposal 2 gives the display rule something to go on: severity says whether a message must be shown. Without it, this would be one more hand-ordered effect.
- Which input a batch came from is known in `main.rs`, where SDL and Launchkey events are dispatched separately. The rule could live in the effect runner if `dispatch` were told the source, or in `main.rs` after dispatch, by reading the batch's current message. The second option touches less.
- Hypothesis (unverified on hardware): the display shows roughly 16–20 characters per line. Messages that go to it need a short first line, and the full text stays in the terminal.
- Hypothesis (unverified on hardware): text written to the DAW-mode display target stays until replaced. If so, an Info shown there needs a way to go away, like Info fading on the status line, e.g. by restoring the DAW pad sub-mode label after a few seconds.

## Open questions

- Should the display show anything for keyboard (SDL) actions, or only for controller actions?
- Does the device offer a temporary or overlay display arrangement, or does everything have to go through the DAW-mode target? `DisplayArrangement` in `launchkey.rs` lists unused arrangements (`TitleNameAndText`, `TitleAndEncoderNames`, `NameAndNumber`) that may be worth testing.
