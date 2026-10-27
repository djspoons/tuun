# Proposal 1: Separate user output from trace on stdout

**Status:** Implemented in `a9f9ff8` ("Route internal tracing through log/env_logger").

## Problem

The native app shows the user messages in three places: the status line at the bottom of the GUI, stdout, and the Launchkey display. The design treats the status line as a one-line summary and the terminal as the place for details (an error's full source snippet, module type errors, `debug()` output, `PrintEvaluatedProgram`).

That only works if stdout is readable, and it wasn't. User-facing output was mixed with developer trace that nobody had asked for:

- The tracker printed every command it received with the full waveform, including a `Modify` for every slider flush (about once per audio buffer while a slider moves). It did this on the audio thread.
- Precompute printed `Precomputing output for …` and `precompute returned: …` on every play.
- Every type check printed `Checking programs... N total errors in …s`.
- The Launchkey decoder printed every controller and note message, plus "Ignoring event …" lines.
- Batch mode printed `Still running, N waveforms remaining` once per buffer, and the tracker clock resync once per buffer.

The only place module type errors appeared was stdout, buried among these lines.

## Proposal

Route trace through the `log` crate, and keep user-facing output as plain `println!`.

- `log` records carry their module path as their target, so the components are just the modules: `tuun::tracker`, `tuun::generator` (precompute), `tuun::environment` (type-check timing), `tuun::launchkey`. There's no component enum or registry to maintain.
- The native binary installs `env_logger` with a default filter of `warn`. Tracing is turned on per component with `RUST_LOG`, e.g. `RUST_LOG=tuun::tracker=debug`.
- Levels:
  - `trace`: bulky dumps (the tracker's full new waveform after a Modify).
  - `debug`: one line per event (tracker commands, precompute progress, check timing, MIDI decoding, unhandled SDL events).
  - `info`: occasional status (tracker clock resync, audio spec).
  - `warn` / `error`: real anomalies, shown by default (missed repetitions, Launchkey send failures, out-of-range program slots, failed sample writes, generator fallbacks).
- User-facing output stays `println!` and is never filtered: status messages, diagnostics, module errors, `debug()` output, `PrintEvaluatedProgram`, batch-mode "Playing …" lines, and the Launchkey-not-found notice.

Disabled `log` macros check the level before formatting, so the tracker no longer formats whole waveforms on the audio thread unless asked to. Hypothesis (unverified): that formatting could cause audible glitches during slider moves.

Also: `Diagnostic::report()` gives the full text of one diagnostic (severity, message, snippet). Batch mode, module findings and `diagnostics::error_message` share it in place of three hand-written loops.

## Why not a `--trace` flag

A single flag would be all-or-nothing, or need its own per-component plumbing. `log`'s per-module filtering costs about the same and needs no extra code.

## Costs

- `log` is a direct dependency. `env_logger` is enabled only by the `native` feature, so the wasm build doesn't pull it in. With its default features it adds `jiff`, `anstream` and `env_filter` to `Cargo.lock`, which provide timestamps and colour.
- Behaviour change: the tracker clock resync and unhandled-SDL-event lines no longer print by default.
- Warn-level lines from the tracker and generator still print on the audio thread when they fire. They are rare anomalies, not per-buffer chatter.

## Result

Measured in batch mode on `lib/v0/lessons/counterpoint/first.tuun`: default output dropped from 293 lines to 28, all user-facing. `RUST_LOG=tuun::tracker=debug` brings back only the tracker's lines. The README documents `RUST_LOG` and the useful targets.
