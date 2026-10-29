# Design study: splitting MIDI from Launchkey support, and structuring AppState / Action / Effect

Status: proposal (read-only study, no code changed). Line numbers refer to commit `61a2585`.

Two concerns were studied, both separately and together:

1. **MIDI split.** Separate generic MIDI (any keyboard can play a keys instrument) from Launchkey-specific support (encoders, buttons, pads, display, DAW-mode handshake). Goals: support other hardware, and eventually Web MIDI in the WASM build.
2. **AppState / Action / Effect sprawl.** `src/lib/actions.rs` (5915 lines) and `src/lib/effects.rs` (2081 lines) cover a lot of ground with little internal structure.

Four approaches were evaluated: two that address both concerns with one design, and one for each concern on its own. Claims marked "Hypothesis (unverified)" were not checked against code or by running anything.

## Recommendation

1. **Do the MIDI split now** (about 3–4 days). It is small, independent, and immediately lets any keyboard play keys.
2. **Reorganize actions.rs by module and sub-struct** (about 3–5 days, can run in parallel). Don't adopt a new dispatch pattern.
3. **Then move Launchkey state into a controller**, so that core no longer emits device effects. Both "together" designs converge on this step.
4. **Defer the portable core `Session`** until there is a concrete plan to run the full app in the browser. Web MIDI keys is blocked mainly on porting keys and voices to wasm, not on the MIDI layering.

## Findings about the current code

### MIDI and Launchkey

- **No MIDI keyboard works without a Launchkey.** `Launchkey::new` (`launchkey.rs:247-324`) returns an error unless it finds the exact port names `"Launchkey MK4 37 DAW In"` and `"Launchkey MK4 37 DAW Out"` (lines 252, 279). It looks for the keys port `"Launchkey MK4 37 MIDI Out"` (line 298) only after both succeed. Other keyboards are never opened, and a different Launchkey model or OS port naming fails too.
- **A malformed MIDI message panics the midir callback thread.** Both decoders call `LiveEvent::parse(message).unwrap()` (`launchkey.rs:514`, `694`).
- **File names don't match their contents.** `midi_input.rs` does no MIDI I/O. It holds the Launchkey event→Action mapping (`classify`, lines 16–149) and the per-frame LED/pad painting (`update_launchkey_state`, lines 151 on). All transport and parsing is in `launchkey.rs`.
- **Notes are part of the Launchkey event enum.** `Event::NoteOn/NoteOff` (`launchkey.rs:81-92`) share an enum with pads and buttons. The keys decoder ignores the channel, and it drops pitch bend, CC (mod wheel, sustain) and aftertouch.
- **Decoder state runs on two threads.** The DAW decoder's `encoder_mode`/`pad_mode`/`shift_held` live on midir's callback thread (`launchkey.rs:45-51`). The main thread keeps its own copy (`Launchkey::encoder_mode`/`pad_mode`), synced through `Effect::SetLaunchkeyEncoderMode`/`SetLaunchkeyPadMode`.
- **The Launchkey mapping depends on the SDL renderer.** `midi_input.rs:8` imports `renderer` for `current_beat_info`/`beat_phase`, and `renderer.rs` depends on sdl2.
- **`midir` is a non-optional dependency** (`Cargo.toml:47`), so the wasm build compiles it as well. The `midi_test` binary uses midir directly and has no `required-features`.
- **Pad LEDs are resent every render frame** (`main.rs:509-517`), without diffing against what the device already shows.
- **Voices are keyed only by the key number** (`WaveformId::Key(u8)`, `player.rs:437`; `note_off_waveforms: HashMap<u8, _>` in `keys.rs`). Hypothesis (unverified): two devices holding the same key would cut each other off.

### WASM

- **`wasm.rs` shares nothing with the app's session code.** It imports only the language modules. It defines its own `MarkId`/`Source` and holds a single waveform. `actions`, `effects`, `keys`, `player`, `programs`, `tracker`, `recorder` and `launchkey` are all `#[cfg(feature = "native")]` (`mod.rs:18-53`). There is no keys instrument or voice handling in the wasm build, so Web MIDI keys needs that port first.
- **Blockers for a portable core** (verified by reading):
  - `tracker.rs` implements `sdl2::audio::AudioCallback` (line 394).
  - `ProgramSet` and `Environment` read files directly.
  - `processing` spawns a thread.
  - `Instant::now()` is called in `messages.rs` and `effects.rs`.

  Hypothesis (unverified): `std::time::Instant::now()` panics on `wasm32-unknown-unknown`, so any shared code must take time as a parameter.
- Hypothesis (unverified): the wasm instance runs inside an AudioWorklet (`web/tuun-processor.js`), where `navigator.requestMIDIAccess` isn't available. The Web MIDI transport would then be JS on the main thread, posting bytes to the worklet.

### AppState / Action / Effect

**Overall shape**

| Item | Size |
|---|---|
| `AppState` | 15 fields (`actions.rs:218-251`) |
| `Action` | 54 variants (372–589) |
| `Effect` | 27 variants (594–711) |
| `apply` | one flat `match` of about 56 arms (718–1139), about 30 of which delegate to helpers |
| Helpers | about 1600 lines (1140–2744); editor/completion is the largest block (1964–2638) |
| Tests | 127, about 3170 lines (54% of the file) |
| `effects.rs::run_one` | a single 370-line `match` (350–720) |

**Guards in `apply`.** Three checks run before the main match:
- Record mode takes transport presses (720–722).
- `refused_by_ownership` (1155–1190).
- `refused_by_processing` (1205–1248).

**Second writer.** The effect runner also writes `AppState`: `keys`, `mode`, `active_program_index`, `programs`, `processing`, `messages`, `waveform_marks` and `should_quit`.

**Coupling runs through the active program's source.**
- About 40 of 54 Actions read or write the active program and are gated on `mode`.
- That set covers the editor, recording, the sequencer, undo, annotations and loop hold. It doesn't line up with any one domain.
- The stale-source check `if ctx.source_stale` is repeated inline at lines 822, 1314, 1603, 1733, 1799, 1932 and 2349. Hypothesis (unverified): the copies use two different texts and severities, an Error "File changed on disk…" and a Warning from `stale_message()`.

**Launchkey share.** Device-specific code is about 10% of the non-test lines:
- 4 of 15 AppState fields: `launch_mode`, `daw_pad_mode`, `sequencer_page`, `loop_hold`. The TODO at line 249 already suggests moving them.
- 7 of 54 Actions: `ChangeSequencerPage`, `SetEncoderMode`, `PadModeChanged`, `ToggleLaunchMode`, `HoldLoop`, `PickLoop`, `ReleaseLoop`.
- 5 of 27 Effects: `SetEncoderDisplay`, `SyncEncoders`, `SetLaunchkeyEncoderMode`, `SetLaunchkeyPadMode`, `SetDawModeDisplay`.

**Core code emits Launchkey effects.** `apply_slider`/`apply_level_db` push `Effect::SetEncoderDisplay` (`actions.rs:2683`, `2729`), and `apply_select_program` emits `SyncEncoders` and resets `loop_hold`. Neither the runner (`effects.rs:619`) nor `Launchkey::set_encoder_display` checks the encoder mode. Hypothesis (unverified): a level change while the encoders are in Plugin mode overwrites a slider's display with "level".

**Hidden misplacements.**
- `Mode::Record { take }` puts recording state inside the editor's mode enum.
- `Program` holds editor undo history, including cursors (`programs.rs:482-528`).
- Messages are written both directly in `apply` and via `Effect::ShowMessage`, and the ordering of message effects matters. So `Effect` has to stay a single ordered `Vec`.

**Launchkey-only features.** Only the Launchkey emits `Transport(_)`, `ToggleProgramPlayback`, `SetSliderNormalized` and `SetLevelDb`; `sdl2_input.rs` never does. Without a Launchkey, for example, you can't record.

## Approach 1: MIDI split only

### Generic layer: `src/lib/midi/`, compiled for every target

`midi/message.rs` depends only on `midly`. It never panics and never reads the clock:

```rust
pub enum MidiMessage {
    NoteOn { channel: u8, key: u8, velocity: u8 },   // velocity > 0
    NoteOff { channel: u8, key: u8 },                // includes velocity-0 NoteOn
    ControlChange { channel: u8, controller: u8, value: u8 },
    PitchBend { channel: u8, value: i16 },
    ChannelPressure { channel: u8, value: u8 },
    SysEx(Vec<u8>),
    Other,
}
pub fn parse(bytes: &[u8]) -> Option<MidiMessage>;
pub trait MidiOutput { fn send(&mut self, bytes: &[u8]); }
```

`midi/keyboard.rs` turns `MidiMessage`s into `KeyboardEvent { NoteOn, NoteOff, PitchBend, ModWheel }`. It handles:
- **Channel filtering**: omni or a single channel.
- **Sustain (CC64)**: note-offs are deferred until the pedal is released, so `Action` doesn't change.
- **Multiple devices**: a per-key count of devices holding it. `NoteOff` is sent only when the last one releases, which avoids the `WaveformId::Key(u8)` collision without touching player or actions.

### Native adapter: `midi/native.rs`

Behind `#[cfg(feature = "native")]`, and the only place that touches midir:
- Port listing and opening.
- `MidirOutput: MidiOutput`.
- A generic `EventSink<T>`, lifted from `launchkey.rs:58-78`.
- Timestamping with `Instant::now()`.

### Launchkey layer: `src/lib/launchkey/`

- `protocol.rs` (pure): constants, the `Event` enum without notes, `EncoderMode`/`PadMode`/`Color`, a `Decoder` (today's `DAWState` without the sink), and byte builders for pads, display and the handshake.
- `device.rs` (native): the `Launchkey` struct over `Box<dyn MidiOutput>`. It claims only the DAW ports, matched by substring and model prefix rather than exact name, and logs each claim.
- `mapping.rs` (native): today's `midi_input::classify` and `update_launchkey_state`, moved unchanged except for the removed note arms.
- `mod.rs` re-exports `EncoderMode`/`PadMode`, so `actions.rs` and `renderer.rs` don't change.

`midi_input.rs` becomes a generic `KeyboardEvent → Action` mapping.

### New devices

- **A generic keyboard** needs no code: the native layer opens every input not claimed by a surface and not a "Through" port. An optional `--midi-in`/`--midi-channel` flag narrows that. The Launchkey's own "MIDI Out" port is just one more keyboard.
- **A knob controller** gets its own mapping file, which maps to the existing `SetSliderNormalized`/`SetLevelDb` actions. Absolute knobs will need soft takeover (pick-up).
- **Don't introduce a `Surface` trait until a second surface exists.**

### Web MIDI later

- JS `onmidimessage` posts bytes to the worklet.
- A `#[wasm_bindgen] fn midi_message(&mut self, port: u32, data: &[u8])` feeds `midi::parse` and the `KeyboardMapper`.
- No new `web-sys` features are needed, because the bytes arrive as `&[u8]`.
- Hypothesis (unverified): Launchkey LEDs over Web MIDI need `requestMIDIAccess({sysex: true})` and a permission prompt.

### Steps (each leaves the build green)

1. Add `midi::message::parse` and use it in both decoders in place of `.unwrap()`.
2. Make `midir` optional under `native`, and give `midi_test` `required-features = ["native"]`.
3. Extract `launchkey/protocol.rs`. Tests feed bytes and use a recording `MidiOutput` stand-in.
4. Add `midi/keyboard.rs` with tests for the channel filter, sustain deferral and the per-key device count.
5. Add `midi/native.rs` (generic `EventSink<T>`, port listing, `MidirOutput`). Switch `Launchkey::new` to use it.
6. Split the input paths: open keyboards independently and remove `Event::NoteOn/NoteOff`. Check on hardware: Launchkey alone, a generic keyboard alone, and both together.
7. Move the Launchkey mapping into `launchkey/mapping.rs`, with its tests.
8. Optional: CLI flags, and passing pitch bend and mod wheel to the keys instrument (a signature change).

Add `midi::` to the wasm test command so the generic layer is checked against the wasm feature set.

### Tradeoffs

- **Auto-opening every port** could grab loopback ports. Skip "Through" ports and add an allow-list flag.
- **Sustain and the device count change what the recorder captures** (deferred note-offs). Decide explicitly that the recorder captures what was heard, and test that.
- **Launchkey types stay in `AppState`/`Action`**: `DawPadMode`, and the encoder/pad-mode actions. Moving them out is Approach 3.
- **Effort**: about 3–4 days including hardware testing.

## Approach 2: AppState / Action / Effect only

Alternatives considered:

| Alternative | Verdict |
|---|---|
| **A. Nested enums + per-domain reducers** (Elm/Redux): `Action::Edit(EditAction)`, `editor::apply(...)` | Leaks. The cross-cutting Actions still need all of `AppState`. Examples: `EnterEditMode` drops pending playback and closes the undo run; `ReleaseLoop` writes the source; `ToggleKeysAnnotation` writes programs, keys and mode; `SelectProgram` resets `loop_hold`; `Undo` belongs to both the editor and the sequencer. It means about 440 mechanical test edits for a modest gain. |
| **B. Sub-structs + a module per domain, flat enums kept** | **Recommended.** Cheap, incremental, and it keeps the exhaustive match and the central guards. |
| **C. Command objects / trait dispatch** | Rejected. It loses the exhaustive match and `matches!`-style tests, scatters the guard lists, and breaks `midi_input::pad_refused`, which calls the guard check on an Action without applying it. It also goes against the repo's preference for simple code. |
| **D. Policy as data + effects grouped by executor** | Take the guard half: one `source_refusal` covering the stale, loading and ownership checks. Group only the 5 Launchkey effects. Keep one ordered `Vec<Effect>`, because message ordering matters. |

Sketch of B:

```rust
pub enum Mode { Select, Edit(EditSession), MoveSliders, Keys, Record(Recording) }
pub struct AppState {
    programs: ProgramSet, active_program_index: usize, mode: Mode, keys: Option<Keys>,
    pads: PadState, // launch_mode, daw_pad_mode, sequencer_page, loop_hold
    // ...
}
// actions.rs keeps Action, Effect, the guards, and apply()'s match as a table of contents:
Action::InsertText(t) => editor::insert_text(state, &t),
```

With `EditSession` as its own struct, `let Mode::Edit(s) = &mut state.mode` can be held while `state.programs` is borrowed, because the fields are disjoint. That removes the repeated re-matching of `Mode::Edit` (e.g. at 2451/2480, 2503/2514).

Steps (each leaves the build green):

1. Extract the pure text/completion helpers that don't need `AppState` (e.g. `token_at`, candidate selection, word boundaries) into their own module, with their tests.
2. `Mode::Edit(EditSession)`: about 80 mechanical edits across actions, sdl2_input, effects and renderer.
3. `Mode::Record(Recording)` and `AppState.pads: PadState`.
4. Move reducer code into `actions/*.rs` one domain per commit (editor, record, loops/annotations, playback/sequencer), each with its tests and a shared `#[cfg(test)] test_support` for fixtures.
5. Combine the guards into `source_refusal`, removing the inline stale checks one at a time. The stale-source tests catch regressions.
6. Optional: `Effect::Launchkey(..)` with a `run_launchkey` executor, and `Action::Edit(..)` for the 11 text/cursor variants plus `Complete`.

Effort: about 3–5 days.

## Approach 3 (together): devices as controllers

Each device gets a controller on the main thread that owns its device state, turns raw input into core `Action`s, reads `&AppState`, and writes its own output.

```rust
pub trait Controller {
    type Input;
    fn handle(&mut self, input: Self::Input, core: &AppState, status: &Status) -> Vec<Action>;
    fn observe(&mut self, core: &AppState);          // after each dispatch: re-sync displays
    fn refresh(&mut self, core: &AppState, status: &Status, now: Instant); // per frame: LEDs
}
pub struct KeysController;                          // any keyboard: notes → NoteOn/NoteOff
pub struct LaunchkeyController<S: MidiSink> {
    out: S, decoder: launchkey::Decoder,
    launch_mode: LaunchMode, daw_pad_mode: DawPadMode, sequencer_page: u8,
    loop_hold: Option<LoopHold>, shown: DisplayCache,
}
```

What moves where:

- **Into the controller:**
  - The 4 Launchkey AppState fields.
  - The 7 Launchkey Actions. `ReleaseLoop` becomes the controller emitting the core `SetLoopAfterBeats`.
  - The 5 Launchkey Effects, replaced by `observe()` diffing a small snapshot against `DisplayCache`. The same diffing throttles the per-frame pad updates.
- **Decoding**: moves to the main thread. The midir callback forwards only bytes and a timestamp, which removes the mode copy. This matches Web MIDI, where messages arrive on the JS main thread.
- **Into an SDL controller (optional)**: `pending_prefix` and the mouse slider.
- **Renderer**: gets a device-neutral knob overlay instead of `launchkey::EncoderMode`.
- **`main.rs`**: holds concrete controller fields, not `dyn` objects.

Tradeoffs:
- **It solves concern 1 but only about 10% of concern 2.** Core still needs Approach 2.
- **The coupling moves rather than disappears.** LED painting reads `mode`, `keys`, `programs` and status, and pad dimming runs `refused_by_ownership` on would-be Actions. Controllers need wide read access and a stable set of public query functions.
- **Updates happen in two phases.** Today `SelectProgram` resets the loop-hold pick in the same reducer step. Afterwards it happens in `observe()`, in a later message batch, unless controller messages go through `dispatch` as `Action::ShowMessage`.
- **Tests need a controller harness**, and device behaviour still has no CI coverage.
- **Effort**: about 1–2 weeks for the MIDI and controller work, plus Approach 2 for core.

## Approach 4 (together): portable core session + surfaces

A platform-independent `Session` (programs, keys, metronome, recording take, the active program, claims, jobs, waveform marks) behind a narrow `Command`/`Event` API. The SDL editor, the Launchkey surface and a future web UI each own their UI state.

```rust
impl Session {
    pub fn handle(&mut self, cx: &Context, ports: &mut dyn Ports, cmd: Command) -> Vec<Event>;
    pub fn finish(&mut self, cx: &Context, ports: &mut dyn Ports, reply: Reply) -> Vec<Event>;
    pub fn view(&self) -> SessionView<'_>;
}
pub trait Ports { fn audio(&mut self) -> &mut dyn AudioSink;
                  fn jobs(&mut self) -> &mut dyn JobQueue;
                  fn storage(&mut self) -> &mut dyn Storage; }
pub trait Surface {
    type Input;
    fn input(&mut self, i: Self::Input, v: &SessionView, out: &mut Vec<Command>);
    fn event(&mut self, e: &Event, v: &SessionView, out: &mut Vec<Command>);
    fn present(&mut self, v: &SessionView, status: &TransportStatus, now: Tick);
}
```

### How today's code would be split

| | Core | Editor / shell | Launchkey | Shared or other |
|---|---|---|---|---|
| AppState fields (15) | 5 | 4 | 4 | 2: active program (becomes core `focus`), messages |
| Actions (54) | 15 | 24 | 7 | 7 shared gestures that split into commands, 1 debug |
| Effects (27) | 16 | 5 | 5 | 1 (`ShowMessage`) |

### What already exists

- `Context` already injects `now`, status, tempo and `source_stale`.
- `World` is already a ports struct.
- `Player` already talks to the tracker only through `tracker::Command`.
- Processing already uses `Job`/`Reply` messages.

### Hard parts

- **Draft text is core state.** Edit-mode keystrokes write `Program.text` directly, and the renderer, the pads and `sequencer::analyze` all see uncommitted text.
- **Undo history is shared.** The editor's undo (with cursors), the Launchkey Undo button, sequencer toggles and recording takes all walk the same `Program` history.
- **Evaluation continuations carry editor effects** (`SetMode` in `on_success`/`on_failure`). They should become a `RequestId` plus an `Event::Evaluated`.
- **Editor ownership gates Launchkey input** (`refused_by_ownership`, `held_loop_refusal`). This needs a core claim/lease concept.
- **Portability work**: an injected clock, Storage, a module source, an inline job queue for wasm, and moving the tracker's `AudioCallback` impl into a native adapter.

### Steps

1. Pure file moves; move `current_beat_info`/`beat_phase` out of the renderer.
2. Generic MIDI.
3. Launchkey protocol/device split.
4. Launchkey surface struct.
5. Outbound `Event` stream; `RequestId` replaces `SetMode` continuations.
6. Move the recording take out of `Mode`.
7. Editor draft buffer and claims. This is the hardest step and it changes behaviour: undo across commits, and what other surfaces see mid-edit.
8. Rename: `AppState` becomes `Session` + `EditorState` + `LaunchkeySurface`.
9. Portability ports and removing the `native` gates on the core.
10. Rebuild `Wasm` on `Session` and add Web MIDI.

### Tradeoffs

- **Benefits**: each layer can be tested on its own, a new device is a new `Surface`, and the web reuses the core.
- **Costs**:
  - More types, and a two-level flow (surface → core → events → surfaces).
  - The message batching (`messages.begin_batch`, `effects.rs:110`) moves into the editor.
  - Heavy test churn.
- **Things that get harder**: features that deliberately span surfaces, such as dimming pads while editing.
- **Effort**: about 5–8 weeks. Steps 1–5 deliver most of the value for both concerns, and steps 9–10 can wait.

## Comparison

| Approach | Fixes MIDI split | Fixes sprawl | Effort | Main risk |
|---|---|---|---|---|
| 1. MIDI split only | Yes | No | ~3–4 days | Opening every port grabs loopback ports |
| 2. AppState reorganization only | No | Mostly, by organization | ~3–5 days | Low |
| 3. Controllers (together) | Yes | ~10% (core still needs 2) | ~1–2 weeks + 2 | Two-phase updates; test harness |
| 4. Core session + surfaces (together) | Yes | Yes, by splitting state three ways | ~5–8 weeks | Redesigning draft text, undo and claims |

### Points the studies agree on

- Make `midir` optional under `native`, and keep the parser pure and free of panics and clock reads.
- Decode Launchkey messages on the main thread, so encoder and pad mode have a single owner.
- Hold off on a `Surface`/`Controller` trait until a second surface exists; concrete fields or an enum are enough.
- Only send pad updates that changed, before targeting Web MIDI.
- Keep the active program shared between the editor and the Launchkey; per-surface focus adds a sync problem for no benefit.

## Open questions

- Should the stale-source refusals share one message and one severity when `source_stale` is combined?
- Should the recorder capture sustain-deferred note-offs as heard?
- Should recording and transport be reachable without a Launchkey, via configurable CCs or SDL keys?
- For Approach 4: should the editor have a draft buffer separate from committed text, and what should undo do across commits?
